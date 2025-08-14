import json
from typing import List
from utils.models.question_model import Question
from datetime import datetime

def extract_questions_from_response(response: str, academy: int,
                                    topic: int, llm_model: str) -> List[Question]:
    try:
        json_data = json.loads(response)
        questions_data = json_data.get("questions", [])
        questions = []
        for q_data in questions_data:
            q_data.setdefault("academy", academy)
            q_data.setdefault("topic", topic)
            q_data.setdefault("llm_model", llm_model)
            q_data.setdefault("createdAt", datetime.now())
            questions.append(Question(**q_data))
        return questions
    except Exception as e:
        print(f"Error parsing questions: {e}")
        return []

def extract_feedback_from_response(response: str) -> list[str]:
    try:
        json_data = json.loads(response)
        return json_data.get("feedbacks", [])
    except Exception:
        return []

def merge_feedback_into_questions(questions: List[Question], feedbacks: list[str]) -> List[Question]:
    for i, question in enumerate(questions):
        print(feedbacks[i])
        retro_text = feedbacks[i] if i < len(feedbacks) else ""
        questions[i] = question.copy_with(tip=retro_text)
    return questions
