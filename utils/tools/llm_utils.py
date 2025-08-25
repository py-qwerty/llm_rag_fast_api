import json
from typing import List
from utils.models.question_model import Question, QuestionList
from datetime import datetime

def extract_questions_from_response(response: list[Question], academy: int,
                                    topic: int, llm_model: str) -> List[Question]:
    try:
        updated_questions = []
        for q in response:
            # Crear una copia con los valores actualizados
            updated_q = Question(
                id=q.id,
                academy=q.academy if q.academy is not None else academy,
                question=q.question,
                answer1=q.answer1,
                answer2=q.answer2,
                answer3=q.answer3,
                answer4=q.answer4,
                solution=q.solution,
                tip=q.tip,
                topic=q.topic if q.topic is not None else topic,
                createdAt=q.createdAt if q.createdAt is not None else datetime.now(),
                order=q.order,
                question_prompt=q.question_prompt,
                llm_model=q.llm_model if q.llm_model is not None else llm_model
            )
            updated_questions.append(updated_q)
        
        return updated_questions
    except Exception as e:
        print(f"Error updating questions: {e}")
        return []

def merge_feedback_into_questions(questions: List[Question], feedbacks: list[str]) -> List[Question]:
    for i, question in enumerate(questions):
        
        retro_text = feedbacks[i] if i < len(feedbacks) else ""
        # print(f"Feedback for question {i}: {feedbacks[i]}")
        questions[i] = question.copy_with(tip=retro_text)
    return questions
