import asyncio
import json
import math
from typing import List, Dict
from agents import Runner
from gotrue import List

from utils.models.question_model import Question
from utils.repository.agent_repository import AgentRepository
from utils.repository.supabase_repository import SupabaseRepository
from utils.tools.llm_utils import extract_questions_from_response, merge_feedback_into_questions



class QuestionRepository:
    def __init__(self):
        self.agent_repo = AgentRepository()
        self.agent = self.agent_repo.agent
        self.runner = Runner()
        self.chunkAgent = self.agent_repo.chunkAgent()

    async def generate_questions_with_feedback(
        self,
        topic: int,
        prompt: str,
        context: str | None,
        academy: int,
        has4questions: bool,
        num_of_q: int,
        llm_model: str,
        max_tokens_per_chunk: int = 100,
        batch_size: int = 30,  # Nuevo parámetro para activar/desactivar RAG # Número de documentos más similares a recuperar
    ) -> list[Question] | str:
        try:
            SBClient = SupabaseRepository()

            # Obtener orden inicial
            current_order = self._get_current_order(SBClient, topic)

            # Chunkear contexto
            chunks, use_context_chunks = await self._chunk_context(
                context, max_tokens_per_chunk, batch_size
            )

            # Preparar prompts de generación de preguntas
            parallel_prompts = self._generate_question_prompts(
                prompt, chunks, use_context_chunks,
                num_of_q, batch_size,
                academy, topic, llm_model, has4questions
            )

            # Ejecutar y procesar respuestas de preguntas
            questions = await self._process_question_responses(
                parallel_prompts, current_order,
                academy, topic, llm_model
            )

            # Generar feedback
            if questions:
                questions_with_feedback = await self._generate_feedback(
                    questions, academy, topic, batch_size
                )
                return questions_with_feedback

            print("❌ No se generaron preguntas, devolviendo lista vacía")
            return []

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error en generate_questions_with_feedback: {e}\n{error_details}")
            return f"Error al generar preguntas y feedback: {e}"

    # 🔹 Nuevo método para procesar contexto con RAG

    def _combine_contexts(self, original_context: str, similar_docs: List[Dict]) -> str:
        """
        Combina el contexto original con los documentos similares encontrados
        """
        combined_context = f"CONTEXTO ORIGINAL:\n{original_context}\n\n"
        combined_context += "INFORMACIÓN RELACIONADA ENCONTRADA:\n\n"
        
        for i, doc in enumerate(similar_docs, 1):
            similarity_score = doc.get('similarity', 0)
            content = doc.get('content', '')
            source = doc.get('source', f'Documento {i}')
            
            combined_context += f"--- Fuente {i}: {source} (similitud: {similarity_score:.3f}) ---\n"
            combined_context += f"{content}\n\n"
        
        return combined_context



    # 🔹 Métodos auxiliares existentes (sin cambios)
    def _get_current_order(self, SBClient, topic: int) -> int:
        last_order_data = SBClient.select(
            "questions",
            filters={"topic": topic},
            order_by="order",
            order_dir="desc",
            limit=1
        )
        return (last_order_data[0]["order"] if last_order_data else 0) + 1

    async def _chunk_context(self, context: str | None, max_tokens: int, batch_size: int) -> tuple[list[str], bool]:
        if not context or not context.strip():
            print("⚠️ No hay contexto, se generarán preguntas solo con el prompt.")
            return [], False

        context = context.strip()
        context_length = len(context)
        
        # Estimar tokens aproximados (1 token ≈ 4 caracteres para texto en español/inglés)
        estimated_tokens = context_length // 4
        
        print(f"📊 Analizando contexto: {context_length} chars (~{estimated_tokens} tokens)")
        
        # Si el contexto es pequeño, no hace falta chunkearlo
        if estimated_tokens <= max_tokens * 1.2:  # 20% de margen de seguridad
            print("✅ Contexto pequeño, no requiere chunkeo")
            return [context], True
        
        print(f"📊 Contexto largo detectado, iniciando chunkeo paralelo...")
        sections_for_chunking = min(batch_size, max(1, context_length // 2000))

        if sections_for_chunking == 1:
            # Chunkeo simple
            chunk_response = await self.runner.run(self.chunkAgent, f"""
            Devuelve este contexto en chunks que tengan longitud máxima de {max_tokens} tokens:

            {context}

            IMPORTANTE: Devuelve EXACTAMENTE este formato JSON:
            {{
                "chunks": ["chunk 1", "chunk 2"]
            }}
            """)
            chunks = chunk_response.final_output_as(list[str])
            print(f"✅ Chunkeo simple completado: {len(chunks)} chunks generados")
            return chunks, True

        # Chunkeo paralelo para contextos muy largos
        section_size = context_length // sections_for_chunking
        chunk_prompts = []
        for i in range(sections_for_chunking):
            start_pos = i * section_size
            section_context = context[start_pos:] if i == sections_for_chunking - 1 else context[start_pos:(i + 1) * section_size]
            chunk_prompt = f"""
            Devuelve esta sección de contexto dividida en chunks de máximo {max_tokens} tokens:

            {section_context}

            IMPORTANTE: Devuelve EXACTAMENTE este formato JSON:
            {{
                "chunks": ["chunk 1", "chunk 2"]
            }}
            """
            chunk_prompts.append(chunk_prompt)

        print(f"🚀 Ejecutando {sections_for_chunking} agentes de chunkeo en paralelo...")
        chunk_responses = await asyncio.gather(
            *[self.runner.run(self.chunkAgent, prompt) for prompt in chunk_prompts],
            return_exceptions=True
        )

        chunks: list[str] = []
        for i, response in enumerate(chunk_responses):
            if isinstance(response, Exception):
                print(f"❌ Error en chunkeo de sección {i}: {response}")
                continue
            try:
                section_chunks = response.final_output_as(list[str])
                chunks.extend(section_chunks)
            except Exception as e:
                print(f"❌ Error procesando chunks de sección {i}: {e}")
        
        print(f"✅ Chunkeo paralelo completado: {len(chunks)} chunks totales generados")
        return chunks, True

    def _generate_question_prompts(
        self, prompt: str, chunks: list[str], use_chunks: bool,
        num_of_q: int, batch_size: int,
        academy: int, topic: int, llm_model: str, has4questions: bool
    ) -> list[str]:
        num_parallel = min(batch_size, num_of_q)
        per_exec, extra = divmod(num_of_q, num_parallel)

        prompts = []
        for i in range(num_parallel):
            q_count = per_exec + (1 if i < extra else 0)
            if use_chunks:
                chunk = chunks[i % len(chunks)]
                question_prompt = f"""{prompt}
                Genera {q_count} preguntas basadas en este texto:

                "{chunk}"

                Devuelve EXACTAMENTE este formato JSON:
                {{
                    "questions": [
                        {{
                            "academy": "{academy}",
                            "question": "Texto de la pregunta aquí",
                            "answer1": "Primera opción",
                            "answer2": "Segunda opción",
                            "answer3": "Tercera opción",
                            "answer4": {"'Cuarta opción'" if has4questions else "null"},
                            "solution": 1,
                            "tip": "Consejo opcional",
                            "topic": {topic},
                            "question_prompt": "{chunk[:200]}...",
                            "retro_text": "",
                            "by_llm": true,
                            "llm_model": "{llm_model}"
                        }}
                    ]
                }}"""
            else:
                question_prompt = f"""{prompt}
                Genera {q_count} preguntas basadas únicamente en el tema.

                Devuelve EXACTAMENTE este formato JSON:
                {{
                    "questions": [
                        {{
                            "academy": "{academy}",
                            "question": "Texto de la pregunta aquí",
                            "answer1": "Primera opción",
                            "answer2": "Segunda opción",
                            "answer3": "Tercera opción",
                            "answer4": {"'Cuarta opción'" if has4questions else "null"},
                            "solution": 1,
                            "tip": "Consejo opcional",
                            "topic": {topic},
                            "question_prompt": "{prompt[:200]}...",
                            "retro_text": "",
                            "by_llm": true,
                            "llm_model": "{llm_model}"
                        }}
                    ]
                }}"""
            prompts.append(question_prompt)
        return prompts

    async def _process_question_responses(self, prompts, current_order, academy, topic, llm_model):
        print(f"🚀 Ejecutando {len(prompts)} agentes en paralelo...")
        responses = await asyncio.gather(
            *[self.runner.run(self.agent, prompt) for prompt in prompts],
            return_exceptions=True
        )

        questions: list[Question] = []
        for response in responses:
            if isinstance(response, Exception):
                continue
            try:
                chunk_questions = extract_questions_from_response(
                    response.final_output_as(list[Question]),
                    academy, topic, llm_model
                )
                for q in chunk_questions:
                    q.order = current_order
                    current_order += 1
                questions.extend(chunk_questions)
            except Exception as e:
                print(f"❌ Error procesando respuesta: {e}")
        return questions

    async def _generate_feedback(self, questions: list[Question], academy: int, topic: int, batch_size: int) -> list[Question]:
        print(f"🔄 Generando feedback para {len(questions)} preguntas...")
        num_agents = min(batch_size, len(questions))
        per_agent, extra = divmod(len(questions), num_agents)

        feedback_prompts = []
        start = 0
        for i in range(num_agents):
            count = per_agent + (1 if i < extra else 0)
            subset = questions[start:start+count]
            subset_dict = [q.model_dump() for q in subset]
            feedback_prompts.append(f"""
            Analiza estas preguntas para el topic {topic} y academia {academy}:

            {json.dumps(subset_dict, ensure_ascii=False, indent=2, default=str)}

            Devuelve estrictamente un JSON:
            {{
                "feedbacks": ["Explicación 1...", "Explicación 2..."]
            }}
            """)
            start += count

        responses = await asyncio.gather(
            *[self.runner.run(self.agent, p) for p in feedback_prompts],
            return_exceptions=True
        )

        all_feedbacks = []
        for response in responses:
            if isinstance(response, Exception):
                continue
            try:
                all_feedbacks.extend(response.final_output_as(list[str]))
            except Exception as e:
                print(f"❌ Error procesando feedback: {e}")

        return merge_feedback_into_questions(questions, all_feedbacks)