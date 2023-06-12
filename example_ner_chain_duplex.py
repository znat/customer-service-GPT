import yaml
from langchain.chat_models import ChatOpenAI
from rich.console import Console
from rich.prompt import Prompt

from lib.ner.entities.basic_entities import (BooleanEntity, EmailEntity,
                                             Entity, EntityExample, IntEntity)
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.ner.ner_chain import NERChain

entity_extractor_llm = ChatOpenAI(
    temperature=0, client=None, max_tokens=256, model="gpt-3.5-turbo"
)

ner_chain = NERChain(
    llm=entity_extractor_llm,
    verbose=True,
    entities={
        "availability": DateTimeEntity,
        "confirmation": BooleanEntity,
    }, 
    examples=[
        EntityExample.parse_obj(e)
        for e in yaml.safe_load(open("duplex_bot_entity_examples.yaml"))
    ],
)

import sys

if __name__ == "__main__":
    console = Console()
    if len(sys.argv) > 2:
        input = sys.argv[1]
        context = sys.argv[2]
    else:
        input = Prompt.ask("Input")
        context = Prompt.ask("Context")
    entities = ner_chain.run(
        {
            "input": input,
            "history": f"User: Hello\nAI: {context}",
        }
    )
    console.print(entities)