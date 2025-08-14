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
            output_type=list[Question],
            
        )

    def feedbackAgent(self):
        SBClient = SupabaseRepository()
        prompt_data = SBClient.select("gpt_prompts", {"destination": "feedback"})
        instructions = prompt_data[0].get("prompt_system", "") if prompt_data else ""
        return Agent(
            name="Analizador de Feedback",
            handoff_description="Un tutor de la Policia Nacional que proporciona retroalimentación sobre las preguntas.",
            instructions=instructions,
            output_type=list[str]
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

    def spanishConstitutionAgent(self):
        return Agent(
            name="Agente de la Constitución Española",
            handoff_description="Un agente que proporciona información sobre la Constitución Española.",
            instructions=(
                "Proporciona información detallada y precisa sobre la Constitución Española, "
                "incluyendo sus artículos, derechos y deberes fundamentales."
            ),
            output_type=str,
        )
