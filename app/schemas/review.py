# Схемы для валидации запросов и ответов

from pydantic import BaseModel
from typing import Optional, List, Union

class ReviewCreate(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    id: int
    text: str

    class Config:
        orm_mode = True

class FindSimilarRequest(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str

class StatusResponse(BaseModel):
    status: str
    result: Optional[Union[List[str], str]] = None  # Может быть списком или строкой ошибки