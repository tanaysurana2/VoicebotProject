import requests
import logging
from fastapi import HTTPException
from typing import Dict, List
from models.base_models import InterviewRequestModel

class InterviewAPI:
    def __init__(self, config_service_root: str):
        self.config_service_root = config_service_root
        self.active_sessions: Dict[str, Dict] = {}

    async def fetch_questions(self, request: InterviewRequestModel) -> List[str]:
        try:
            external_api_url = f"{self.config_service_root}/v2/generate_qna"
            logging.info(f"Fetching questions from: {external_api_url}")    
            logging.info(f"Request payload: {request.to_dict()}")
            response = requests.post(external_api_url, json=request.to_dict())
            response.raise_for_status()
            
            qna_data = response.json()
            print(f"qna_data: {qna_data}")
            questions = []
            for q in qna_data.get("questions", []):
                if q.get("question_text"):
                    logging.info(f"Processing question: {q}")
                    question_text = q["question_text"].strip()
                    is_mandatory = q.get("is_mandatory", False)
                    skill = q.get("skill", "")
                    
                    if question_text:
                        if is_mandatory:
                            questions.insert(0, question_text)
                        else:
                            questions.append(question_text)
            
            logging.info(f"Fetched {len(questions)} questions:")
            for i, q in enumerate(questions, 1):
                logging.info(f"Question {i}: {q}")
            
            if not questions:
                logging.error("No questions fetched from API. Using default questions.")
                questions = [
                    "Can you tell me about yourself?",
                    "What are your strengths and weaknesses?",
                    "Where do you see yourself in five years?"
                ]
            
            return questions
        except Exception as e:
            logging.error(f"Error fetching questions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {str(e)}")

    async def get_followup_question(self, question: str, answer: str) -> str:
        try:
            followup_api_url = f"{self.config_service_root}/v3/generate_followup"
            payload = {
                "question": question,
                "answer": answer
            }
            logging.info(f"Generating follow-up question for: {question}")
            logging.info(f"Previous answer: {answer}")
            logging.info(f"Follow-up API request payload: {payload}")
            
            response = requests.post(followup_api_url, json=payload)
            response.raise_for_status()
            
            followup_data = response.json()
            print(f"[DEBUG] Follow-up API raw response: {followup_data}")
            followup_question = followup_data.get("follow_up_question", "")
            
            if not followup_question:
                logging.warning("No follow-up question received from API")
                return None
                
            logging.info(f"Generated follow-up question: {followup_question}")
            return followup_question
            
        except Exception as e:
            logging.error(f"Error generating follow-up question: {str(e)}")
            return None