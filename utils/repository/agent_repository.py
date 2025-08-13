from agents import Agent, Runner
from utils.repository.supabase_repository import SupabaseRepository
from utils.models.question_model import Question, QuestionList
from datetime import datetime
from typing import List
import json
import math

from utils.tools.llm_utils import extract_feedback_from_response, extract_questions_from_response, merge_feedback_into_questions

class AgentRepository:

    def __init__(self, context: str):
        self.context = context
        self.agent = Agent(
            name="Tutor Policia Nacional - Coordinador",
            handoff_description="Un tutor coordinador de la Policia Nacional que determina si generar preguntas o dar feedback.",
            instructions="""Eres un coordinador que decide qué acción tomar basándote en la solicitud del usuario:
            - Si el usuario quiere generar preguntas, transfiere al agente questionAgent
            - Si el usuario quiere feedback o análisis de preguntas, transfiere al agente feedbackAgent
            - Si no está claro, pregunta al usuario qué necesita específicamente""",
            handoffs=[self.questionAgent(), self.feedbackAgent()],
            output_type=QuestionList,
        )
        self.runner = Runner()
    
    def questionAgent(self):
        SBClient = SupabaseRepository()
        prompt_data = SBClient.select("gpt_prompts", {"destination": "generate_question"})
        instructions = prompt_data[0].get("prompt_system", "") if prompt_data else ""
        return Agent(
            name="Generador de Preguntas",
            handoff_description="Un tutor de la Policia Nacional que crea preguntas para exámenes.",
            instructions=instructions,
            
        )

    def feedbackAgent(self):
        SBClient = SupabaseRepository()
        prompt_data = SBClient.select("gpt_prompts", {"destination": "feedback"})
        instructions = prompt_data[0].get("prompt_system", "") if prompt_data else ""
        return Agent(
            name="Analizador de Feedback",
            handoff_description="Un tutor de la Policia Nacional que proporciona retroalimentación sobre las preguntas.",
            instructions=instructions,
        )

    def chunkAgent(self):

        return Agent(
            name="Chunking Agent",
            handoff_description="Un agente para chunkear un texto en partes más pequeñas.",
            instructions=(
                "Divide el texto para que se puedan sacar varias preguntas de él. "
                "Cada parte debe tener un tamaño máximo de 1000 palabras y los trozos deben ser coherentes. "
                "Devuelve siempre una lista de strings, donde cada string es un chunk generado."
            ),
            output_type=list[str],
            # tools=[self.smart_chunk_text]
        )

    # =====================
    # NUEVAS FUNCIONES PARA CHUNKING
    # =====================

    

    # async def generate_questions_with_feedback(
    #     self,
    #     topic: int,
    #     prompt: str,
    #     academy: int,
    #     has4questions: bool,
    #     num_of_q: int,
    #     llm_model: str,
    #     max_tokens_per_chunk: int = 100
    # ) -> dict | str:
    #     """
    #     Genera preguntas con feedback a partir de un prompt largo.
    #     Maneja chunking si el prompt es demasiado largo.
    #     Cada pregunta tiene 'question_prompt' para rastrear el fragmento de origen.
    #     """
    #     try:
    #         SBClient = SupabaseRepository()

    #         # 0. Obtener el último order asignado para este topic
    #         last_order_data = SBClient.select(
    #             "questions",
    #             filters={"topic": topic},
    #             order_by="order",
    #             order_dir="desc",
    #             limit=1
    #         )
    #         current_order = (last_order_data[0]["order"] if last_order_data else 0) + 1

    #         # 1. Dividir prompt en chunks

    #         chunk_response = await self.runner.run(self.chunkAgent(), f"""

    #         Devuelve este contexto en chunks que tengan longitud máxima de {max_tokens_per_chunk} tokens:

    #         {self.context}

    #         IMPORTANTE: Devuelve EXACTAMENTE este formato JSON:
    #             {{
    #                 "chunks": [
    #                     "chunk 1",
    #                     "chunk 2"
    #                 ]
    #             }}

    #         """)
    #         chunks = chunk_response.final_output_as(list[str])

    #         print(f"Chunks generados: {chunks}")
    #         # chunks = self.chunk_text(prompt, max_tokens=max_tokens_per_chunk)
    #         questions: List[Question] = []

    #         questions_per_chunk = max(1, math.ceil(num_of_q / len(chunks)))

    #         for chunk in chunks:
    #             question_prompt_text = f"""
    #             Genera {questions_per_chunk} preguntas basadas en el siguiente texto:

    #             "{chunk}"

    #             IMPORTANTE: Devuelve EXACTAMENTE este formato JSON:
    #             {{
    #                 "questions": [
    #                     {{
    #                         "academy": "{academy}",
    #                         "question": "Texto de la pregunta aquí",
    #                         "answer1": "Primera opción",
    #                         "answer2": "Segunda opción", 
    #                         "answer3": "Tercera opción",
    #                         "answer4": {"'Cuarta opción'" if has4questions else "null"},
    #                         "solution": 1,
    #                         "tip": "Consejo opcional",
    #                         "topic": {topic},
    #                         "order": 0,
    #                         "question_prompt": {chunk},
    #                         "retro_text": "",
    #                         "llm_model": "{llm_model}"
    #                     }}
    #                 ]
    #             }}

    #             Transfiere esta tarea al agente questionAgent.
    #             """
    #             # 2. Generar preguntas del chunk
    #             response = await self.runner.run(self.agent, question_prompt_text)
    #             chunk_questions = extract_questions_from_response(
    #                 response.final_output_as(QuestionList),
    #                 academy,
    #                 topic,
    #                 llm_model
    #             )
    #             for q in chunk_questions:
    #                 q.order = current_order
    #                 current_order += 1

    #             questions.extend(chunk_questions)


    #             if len(questions) >= num_of_q:
    #                 questions = questions[:num_of_q]
    #                 break

    #         # 3. Preparar datos para feedback
    #         questions_for_feedback = [q.model_dump() for q in questions]
    #         feedback_prompt = f"""
    #         Analiza las siguientes preguntas generadas para el topic {topic} y academia {academy}:

    #         {json.dumps(questions_for_feedback, ensure_ascii=False, indent=2, default=str)}

    #         Para cada pregunta:
    #         - Explica de forma profesional y objetiva por qué la respuesta correcta es la opción más adecuada.
    #         - Fundamenta tu explicación con criterios claros, técnicos o académicos según corresponda.
    #         - Si es relevante, menciona por qué las demás opciones no son correctas.
    #         - Utiliza un lenguaje formal, preciso y fácil de entender.

    #         Devuelve estrictamente un JSON con esta estructura:
    #         {{
    #             "feedbacks": [
    #                 "Explicación de la pregunta 1...",
    #                 "Explicación de la pregunta 2..."
    #             ]
    #         }}

    #         No incluyas texto adicional fuera del JSON.
    #         Transfiere esta tarea al agente feedbackAgent.
    #         """

    #         # 4. Generar feedback
    #         feedback_response = await self.runner.run(
    #             starting_agent=self.agent,
    #             input=feedback_prompt,
            
    #         )
    #         final_feedback_response = feedback_response.final_output_as(list[str])
    #         feedbacks = extract_feedback_from_response(final_feedback_response)

    #         # 5. Combinar feedback con preguntas
    #         questions_with_feedback = merge_feedback_into_questions(questions, feedbacks)


    #         return questions_with_feedback
    #         return {
    #             "questions": questions_with_feedback,
    #             "feedback": feedbacks,
    #             "metadata": {
    #                 "topic": topic,
    #                 "academy": academy,
    #                 "num_questions": len(questions_with_feedback),
    #                 "has_options": has4questions,
    #                 "generation_method": "chunked_prompt",
    #                 "llm_model": llm_model,
    #                 "generated_at": datetime.now().isoformat()
    #             }
    #         }

    #     except Exception as e:
    #         import traceback
    #         error_details = traceback.format_exc()
    #         print(f"❌ Error en generate_questions_with_feedback: {e}\n{error_details}")
    #         return f"Error al generar preguntas y feedback: {e}"
