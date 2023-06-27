from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere
from rich.console import Console
from rich.prompt import Prompt


from lib.ner.entities.basic_entities import (BooleanEntity, EmailEntity,
                                              Entity, EntityExample, IntEntity)
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.ner.ner_chain import NERChain

entity_extractor_llm = ChatOpenAI(
    temperature=0, client=None, max_tokens=256, model="gpt-3.5-turbo"
)

examples = [
    {"text": "I'm Nathan", "entities": [{"name": "first_name", "value": "Nathan"}]},
    {
        "text": "I'm jenny and I'm 98 yo",
        "entities": [
            {"name": "first_name", "value": "Jenny"},
            {"name": "age", "value": 98},
        ],
    },
    {
        "text": "I'm pete, my age is twenty-five and my email is pete@gmail.com",
        "entities": [
            {"name": "first_name", "value": "Pete"},
            {"name": "age", "value": 25},
            {"name": "email", "value": "pete@gmail.com"},
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

    {
        "context": "Is everything correct?",
        "text": "Yes",
        "entities": [{"name": "confirmed", "value": True}],
    },
    {
        "context": "Do you confirm the information above?",
        "text": "Yes",
        "entities": [{"name": "confirmed", "value": True}],
    },
    {"context": "How old are you?", "text": "Yes", "entities": []},
]

examples = [EntityExample.parse_obj(e) for e in examples]
entities = {
    "availability": DateTimeEntity,
    "age": IntEntity,
    "email": EmailEntity,
    "last_name": Entity,
    "occupation": Entity,
    "confirmed": BooleanEntity,
}

ner_chain = NERChain(
    llm=entity_extractor_llm,
    additional_instructions="""
"confirmation" entity should only be `true` if the gives an explicity confirmation that all the collected information is correct
and not if the user just says "yes" or confirms but asks follow-up questions.
    """,
    verbose=True,
    entities=entities,
    examples=examples,
)

import sys

if __name__ == "__main__":
    console = Console()
    if len(sys.argv) > 2:
        input = sys.argv[1]
        context = sys.argv[2]
    else:
        input = Prompt.ask("Text")
        context = Prompt.ask("Context")
    entities = ner_chain.run(
        {
            "input": input,
            "history": f"User: Hello\nAI: {context}",
        }
    )
    console.print(entities)
