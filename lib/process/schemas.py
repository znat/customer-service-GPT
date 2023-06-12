from typing import ClassVar, Optional, Dict
from pydantic import BaseModel
from enum import Enum

class Process(BaseModel):
    process_description: ClassVar[str] = "The goal of this form is to collect information from you"
    def is_completed(self) -> bool:
        return True
    

class Status(str, Enum):
    completed = "completed"
    failed = "failed"

class Result(BaseModel):
    status: Status
    result: Optional[BaseModel]
    errors: Optional[Dict[str, str]]

    class Config:
        use_enum_values = True