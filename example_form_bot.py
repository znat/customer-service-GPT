from langchain.chat_models import ChatOpenAI
from pydantic import Field, validator
from lib.bot import gradio_bot
from lib.conversation_memory import ConversationMemory

from lib.process.process_chain import ProcessChain
from lib.process.schemas import Process
from lib.ner.entities.basic_entities import BooleanEntity, Entity, EntityExample, IntEntity



examples = [
    {
        "text": "Hey, I'm so excited about this!",
        "entities": [],
        "context": "What is your name?",
    },
    {"text": "I'm Nathan", "entities": [{"name": "first_name", "value": "Nathan"}]},
    {
        "text": "I'm jenny and I'm 98 yo",
        "entities": [
            {"name": "first_name", "value": "Jenny"},
            {"name": "age", "value": 98},
        ],
    },
    {
        "text": "I'm Jo Neville and I'm a plumber",
        "entities": [
            {"name": "first_name", "value": "Jo"},
            {"name": "last_name", "value": "Neville"},
            {"name": "occupation", "value": "plumber"},
        ],
    },
    {"text": "That is correct", "entities": [{"name": "confirmed", "value": True}]},
    {
        "context": "Is everything correct?",
        "text": "No",
        "entities": [{"name": "confirmed", "value": False}],
    },
    {
        "context": "Do you confirm the information above?",
        "text": "Yes",
        "entities": [{"name": "confirmed", "value": True}],
    },
    {"context": "How old are you?", "text": "Yes", "entities": []},
]


class MyForm(Process):
    goal = """
You are a retail bank account executive AI.
Your goal is to collect the required information from the User to open a bank account.
"""
    first_name: str = Field(
        name="First name",
        description="First name of the user, required to open an account",
        question="What is your first name?",
    )
    last_name: str = Field(
        name="Last name",
        description="Last name of the user, required to open an account",
        question="What is your last name?",
    )
    age: int = Field(
        name="Age",
        description="Age of the user, required to open an account",
        question="What is your age?",
    )
    occupation: str = Field(
        name="Occupation",
        description="Occupation of the user, required to establish the user's risk profile",
        question="What is your current occupation?",
    )
    confirmed: bool = Field(
        name="Confirmed",
        description="Whether the user has confirmed their details",
        question="You are {{first_name}} {{last_name}}, you are {{age}} years old and your current occupation is {{occupation}}. Is that correct?"
        "",
        exclude=True,
    )

    @validator("first_name")
    def validate_first_name(cls, v):
        assert v[0].isalpha(), "First name must start with a letter."
        return v.capitalize()

    @validator("last_name")
    def validate_last_name(cls, v):
        assert v[0].isalpha(), "Last name must start with a letter."
        return v.capitalize()

    @validator("age")
    def validate_age(cls, v):
        assert v is None or v >= 18, "Age must be 18 or older"
        return v

    @validator("occupation")
    def validate_occupation(cls, v):
        assert len(v) > 2, "Occupation must be at least 2 character"
        return v.lower()
    
    @validator("confirmed")
    def validate_confirmed(cls, v):
        assert v is True, "Please provide me with the correct information"
        return v


openai_entity_extractor_llm = ChatOpenAI(temperature=0, client=None, max_tokens=256)
openai_chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=256)

form_chain = ProcessChain(
    ner_llm=openai_entity_extractor_llm,
    chat_llm=openai_chat_llm,
    entities={
        "first_name": Entity,
        "age": IntEntity,
        "last_name": Entity,
        "occupation": Entity,
        "confirmed": BooleanEntity,
    },
    entity_examples=[EntityExample.parse_obj(e) for e in examples],
    process=MyForm,
    memory=ConversationMemory(),
    verbose=True,
)

gradio_bot(form_chain, "hey").launch()
