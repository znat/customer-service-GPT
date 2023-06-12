from typing import Any, Optional, Type
from langchain import PromptTemplate
from pydantic import BaseModel, Field, validator
from langchain.base_language import BaseLanguageModel
import re


class Entity(BaseModel):
    name: str
    description: Optional[str] = Field(exclude=True, default=None)
    llm: Optional[BaseLanguageModel] = Field(exclude=True, default=None)
    value: Any
    prompt: Optional[PromptTemplate] = Field(exclude=True, default=None)

class BooleanEntity(Entity):
    value: bool

class StringEntity(Entity):
    value: str
    
class EmailEntity(Entity):
    @validator("value")
    def validate_email(cls, v):
        print("email val")
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        match = re.search(pattern, v)
        if not match:
            return None
        return v


class IntEntity(Entity):
    @validator("value")
    def validate_int(cls, v):
        try:
            return int(v)
        except ValueError:
            return None


class EntityExample(BaseModel):
    text: str
    context: Optional[str] = None
    entities: list[Entity]
