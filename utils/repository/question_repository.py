import asyncio
import json
import math
from agents import Runner
from gotrue import List

from utils.models.question_model import Question, QuestionList
from utils.repository.agent_repository import AgentRepository
from utils.repository.supabase_repository import SupabaseRepository
from utils.tools.llm_utils import extract_feedback_from_response, extract_questions_from_response, merge_feedback_into_questions


class QuestionRepository:
    def __init__(self, context: str):

        self.agent_repo = AgentRepository(context=context)
        self.agent = self.agent_repo.agent
        self.runner = Runner()
        self.chunkAgent = self.agent_repo.chunkAgent()
        # self.feedbackAgent = self.agent.feedbackAgent()
        # self.questionAgent = self.agent.questionAgent()

    async def generate_questions_with_feedback(
        self,
        topic: int,
        prompt: str,
        context: str,
        academy: int,
        has4questions: bool,
        num_of_q: int,
        llm_model: str,
        max_tokens_per_chunk: int = 100,
        batch_size: int = 3
    ) -> list[Question] | str:
        try:
            SBClient = SupabaseRepository()

            # Obtener último orden
            last_order_data = SBClient.select(
                "questions",
                filters={"topic": topic},
                order_by="order",
                order_dir="desc",
                limit=1
            )
            current_order = (last_order_data[0]["order"] if last_order_data else 0) + 1

            # Dividir contexto en chunks
            print(f"📊 Iniciando chunkeo paralelo del contexto ({len(context)} chars)...")
        
            # Dividir el contexto en secciones para procesamiento paralelo
            context_length = len(context)
            sections_for_chunking = min(batch_size, max(1, context_length // 2000))  # Al menos 2000 chars por sección
            
            if sections_for_chunking == 1:
                # Si el contexto es pequeño, usar chunkeo normal
                chunk_response = await self.runner.run(self.chunkAgent, f"""
                Devuelve este contexto en chunks que tengan longitud máxima de {max_tokens_per_chunk} tokens:

                {context}

                IMPORTANTE: Devuelve EXACTAMENTE este formato JSON:
                {{
                    "chunks": [
                        "chunk 1",
                        "chunk 2"
                    ]
                }}
                """)
                chunks = chunk_response.final_output_as(list[str])
                print(f"✅ Chunkeo simple completado: {len(chunks)} chunks generados")
            else:
                # Dividir contexto en secciones para chunkeo paralelo
                section_size = context_length // sections_for_chunking
                chunk_prompts = []
                
                for i in range(sections_for_chunking):
                    start_pos = i * section_size
                    if i == sections_for_chunking - 1:  # Última sección incluye el resto
                        section_context = context[start_pos:]
                    else:
                        # Buscar un punto de corte natural (espacio, punto, salto de línea)
                        end_pos = (i + 1) * section_size
                        natural_break = context.rfind(' ', start_pos, end_pos + 100)  # Buscar espacio en los próximos 100 chars
                        if natural_break > start_pos:
                            end_pos = natural_break
                        section_context = context[start_pos:end_pos]
                    
                    chunk_prompt = f"""
                    Devuelve esta sección de contexto dividida en chunks que tengan longitud máxima de {max_tokens_per_chunk} tokens:

                    {section_context}

                    IMPORTANTE: Devuelve EXACTAMENTE este formato JSON:
                    {{
                        "chunks": [
                            "chunk 1",
                            "chunk 2"
                        ]
                    }}
                    """
                    chunk_prompts.append(chunk_prompt)
                    print(f"📝 Sección {i}: {len(section_context)} chars para chunkeo")

                print(f"🚀 Ejecutando {sections_for_chunking} agentes de chunkeo en paralelo...")
                
                # Ejecutar chunkeo en paralelo
                chunk_responses = await asyncio.gather(
                    *[self.runner.run(self.chunkAgent, prompt) for prompt in chunk_prompts],
                    return_exceptions=True
                )

                # Combinar todos los chunks
                chunks = []
                for i, response in enumerate(chunk_responses):
                    try:
                        if isinstance(response, Exception):
                            print(f"❌ Error en chunkeo de sección {i}: {response}")
                            continue
                            
                        section_chunks = response.final_output_as(list[str])
                        chunks.extend(section_chunks)
                        print(f"✅ Sección {i}: {len(section_chunks)} chunks generados")
                        
                    except Exception as e:
                        print(f"❌ Error procesando chunks de sección {i}: {e}")
                        continue

                print(f"✅ Chunkeo paralelo completado: {len(chunks)} chunks totales generados")


            # Determinar cuántas ejecuciones paralelas necesitamos
            # Usamos batch_size como el número máximo de agentes en paralelo
            num_parallel_executions = min(batch_size, num_of_q)
            
            # Calcular preguntas por ejecución
            questions_per_execution = num_of_q // num_parallel_executions
            extra_questions = num_of_q % num_parallel_executions

            # Crear prompts para ejecutar en paralelo
            parallel_prompts = []
            
            for i in range(num_parallel_executions):
                # Determinar cuántas preguntas debe generar esta ejecución
                questions_for_this_execution = questions_per_execution
                if i < extra_questions:  # Distribuir preguntas extra
                    questions_for_this_execution += 1
                
                # Asignar chunk específico para este agente (rotación si hay más agentes que chunks)
                chunk_index = i % len(chunks)
                assigned_chunk = chunks[chunk_index]
                
                question_prompt_text = f"""
                {prompt}
                Genera {questions_for_this_execution} preguntas basadas en el siguiente texto específico:

                "{assigned_chunk}"

                IMPORTANTE: 
                - Enfócate ÚNICAMENTE en el contenido del texto proporcionado
                - Genera preguntas variadas y diferentes entre sí
                - Asegúrate de que cada pregunta aborde aspectos distintos del texto
                
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
                            "question_prompt": "{assigned_chunk[:200]}...",
                            "retro_text": "",
                            "by_llm": true,
                            "llm_model": "{llm_model}"
                        }}
                    ]
                }}

                Transfiere esta tarea al agente questionAgent.
                """
                parallel_prompts.append(question_prompt_text)
                print(f"📝 Agente {i}: asignado chunk {chunk_index} ({len(assigned_chunk)} chars) para generar {questions_for_this_execution} preguntas")


            # Ejecutar todos los prompts en paralelo
            print(f"🚀 Ejecutando {num_parallel_executions} agentes en paralelo para generar {num_of_q} preguntas...")
            
            responses = await asyncio.gather(
                *[self.runner.run(self.agent, prompt) for prompt in parallel_prompts],
                return_exceptions=True
            )

            # Procesar respuestas y recopilar preguntas
            questions: list[Question] = []
            
            for i, response in enumerate(responses):
                try:
                    if isinstance(response, Exception):
                        print(f"❌ Error en ejecución paralela {i}: {response}")
                        continue
                    chunk_questions = extract_questions_from_response(
                        response.final_output_as(list[Question]),
                        academy,
                        topic,
                        llm_model
                    )
                    
                    # Asignar orden secuencial
                    for q in chunk_questions:
                        q.order = current_order
                        current_order += 1
                    
                    questions.extend(chunk_questions)
                    print(f"✅ Ejecución {i}: {len(chunk_questions)} preguntas generadas")
                    
                except Exception as e:
                    print(f"❌ Error procesando respuesta de ejecución {i}: {e}")
                    continue

            # Limitar al número exacto solicitado
            # questions = questions[:num_of_q]
            print(f"📊 Total de preguntas generadas: {len(questions)}/{num_of_q}")

            # Si no tenemos suficientes preguntas, mostrar advertencia
            if len(questions) < num_of_q:
                print(f"⚠️ Solo se generaron {len(questions)} de {num_of_q} preguntas solicitadas")

            # Generar feedback en paralelo si hay preguntas
            if questions:
                print(f"🔄 Generando feedback para {len(questions)} preguntas...")
                
                # Determinar cuántos agentes usar para feedback (mismo que para preguntas)
                num_feedback_agents = min(batch_size, len(questions))
                
                # Dividir preguntas entre agentes de feedback
                questions_per_feedback_agent = len(questions) // num_feedback_agents
                extra_questions_feedback = len(questions) % num_feedback_agents
                
                feedback_prompts = []
                start_idx = 0
                
                for i in range(num_feedback_agents):
                    # Calcular cuántas preguntas debe procesar este agente
                    questions_for_this_agent = questions_per_feedback_agent
                    if i < extra_questions_feedback:
                        questions_for_this_agent += 1
                    
                    # Obtener el subset de preguntas para este agente
                    end_idx = start_idx + questions_for_this_agent
                    questions_subset = questions[start_idx:end_idx]
                    questions_subset_dict = [q.model_dump() for q in questions_subset]
                    
                    feedback_prompt = f"""
                    Analiza las siguientes {len(questions_subset)} preguntas generadas para el topic {topic} y academia {academy}:

                    {json.dumps(questions_subset_dict, ensure_ascii=False, indent=2, default=str)}

                    Para cada pregunta:
                    - Explica de forma profesional y objetiva por qué la respuesta correcta es la opción más adecuada.
                    - Fundamenta tu explicación con criterios claros, técnicos o académicos según corresponda.
                    - Si es relevante, menciona por qué las demás opciones no son correctas.
                    - Utiliza un lenguaje formal, preciso y fácil de entender.
                    - No hagas la típica explicación 'Para la pregunta 1...'

                    Devuelve estrictamente un JSON con esta estructura:
                    {{
                        "feedbacks": [
                            "Explicación de la pregunta 1...",
                            "Explicación de la pregunta 2..."
                        ]
                    }}

                    No incluyas texto adicional fuera del JSON.
                    Transfiere esta tarea al agente feedbackAgent.
                    """
                    
                    feedback_prompts.append(feedback_prompt)
                    start_idx = end_idx

                print(f"🚀 Ejecutando {num_feedback_agents} agentes de feedback en paralelo...")
                
                # Ejecutar feedback en paralelo
                feedback_responses = await asyncio.gather(
                    *[self.runner.run(self.agent, prompt) for prompt in feedback_prompts],
                    return_exceptions=True
                )

                # Procesar respuestas de feedback
                all_feedbacks = []
                
                for i, response in enumerate(feedback_responses):
                    try:
                        if isinstance(response, Exception):
                            print(f"❌ Error en agente de feedback {i}: {response}")
                            continue
                        print(f"response: {response}")
                        final_feedback_response = response.final_output_as(list[str])
                        # print('feedback_respuesas',final_feedback_response)
                        feedbacks = final_feedback_response
                        # print('feedbacks', feedbacks)
                        all_feedbacks.extend(final_feedback_response)
                        print(f"✅ Agente feedback {i}: {len(feedbacks)} feedbacks generados")
                        
                    except Exception as e:
                        print(f"❌ Error procesando feedback del agente {i}: {e}")
                        continue

                # Combinar feedback con preguntas
                questions_with_feedback = merge_feedback_into_questions(questions, all_feedbacks)
                
                print(f"✅ Feedback total generado: {len(all_feedbacks)} feedbacks")
                return questions_with_feedback
            else:
                print("❌ No se generaron preguntas, devolviendo lista vacía")
                return []

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error en generate_questions_with_feedback: {e}\n{error_details}")
            return f"Error al generar preguntas y feedback: {e}"
