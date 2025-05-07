from pydantic import BaseModel
from typing import List, Optional

class InterviewPayload(BaseModel):
    resume_url: str
    job_description: str
    job_application_id: int
    job_skills: List[str]
    mandatory_skills: List[List[str]]

class InterviewRequestModel(BaseModel):
    resume_url: str
    job_description: str = ""
    job_application_id: int
    job_skills: List[str]
    mandatory_skills: List[List[str]]
    optional_skills: List[List[str]] = []

    def to_dict(self) -> dict:
        return {
            "resume_url": self.resume_url,
            "job_description": self.job_description,
            "job_application_id": self.job_application_id,
            "job_skills": self.job_skills,
            "mandatory_skills": self.mandatory_skills,
            "optional_skills": self.optional_skills
        }