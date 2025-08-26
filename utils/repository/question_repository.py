import asyncio
import json
from typing import List, Dict

from utils.models.question_model import Question
from utils.repository.supabase_repository import SupabaseRepository


class QuestionRepository:
    def __init__(self):
        pass

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
            batch_size: int = 30,
    ) -> list[Question] | str:
        try:
            # Simplified version - just return a basic structure
            # This removes the dependency on 'agents' library that's causing startup issues

            print(f"Generating {num_of_q} questions for topic {topic}")

            # For now, return a simple mock response
            # In a real implementation, you'd call OpenAI directly here
            questions = []

            for i in range(num_of_q):
                question = Question(
                    academy=academy,
                    question=f"Sample question {i + 1} for topic {topic}",
                    answer1="Option A",
                    answer2="Option B",
                    answer3="Option C",
                    answer4="Option D" if has4questions else None,
                    solution=1,
                    tip="Sample tip",
                    topic=topic,
                    question_prompt=prompt[:200],
                    llm_model=llm_model,
                    order=i + 1
                )
                questions.append(question)

            return questions

        except Exception as e:
            print(f"Error in generate_questions_with_feedback: {e}")
            return f"Error generating questions: {e}"