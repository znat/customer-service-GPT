from typing import Optional
from pydantic import Field, validator
from lib.process.schemas import Process


class SimpleForm(Process):
    process_description = """
You are a retail bank account executive AI.
Your goal is to collect the required information from the User to open a bank account.
"""
    first_name: Optional[str] = Field(
        name="First name",
        description="First name of the user, required to open an account",
        question="What is your first name?",
    )

    age: Optional[int] = Field(
        name="Age",
        description="Age of the user, required to open an account",
        question="What is your age?",
    )

    @validator("first_name")
    def validate_first_name(cls, v):
        assert v[0].isalpha(), "First name must start with a letter."
        return v.capitalize()

    @validator("age")
    def validate_age(cls, v):
        assert v is None or v >= 18, "Age must be 18 or older"
        return v