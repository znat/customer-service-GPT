from typing import ClassVar, Optional, Dict, Union, get_args, get_origin
from pydantic import BaseModel, Field, root_validator, validator
from enum import Enum

class Process(BaseModel):
    process_description: ClassVar[str] = "The goal of this form is to collect information from you"

    
    def is_completed(self) -> bool:
        return True
    
    @root_validator(pre=True)
    def all_fields_optional(cls, values):
        for field_name, field_value in values.items():
            field_type = cls.__annotations__[field_name]
            if not get_origin(field_type) is Union or not type(None) in get_args(field_type):
                raise ValueError(f"All Process fields must be Optional, and {field_name} was not")
        return values

class Status(str, Enum):
    completed = "completed"
    failed = "failed"

class Result(BaseModel):
    status: Status
    result: Optional[BaseModel]
    errors: Optional[Dict[str, str]]

    class Config:
        use_enum_values = True

class MyProcess(Process):
    name: str = Field(question="What is your name")


