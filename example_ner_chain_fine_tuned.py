from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere
from rich.console import Console
from rich.prompt import Prompt

from lib.ner.entities.basic_entities import (BooleanEntity, EmailEntity,
                                              Entity, EntityExample, IntEntity)
from lib.ner.ner_chain import NERChain

entity_extractor_llm = ChatOpenAI(
    temperature=0, client=None, max_tokens=256, model="gpt-3.5-turbo"
)
cohere_ner_llm = Cohere(model="776e218a-7bd9-44fc-a405-3b58f134bd8f-ft")


entities = {
    "availability": Entity,
    "last_name": Entity,
    "first_name": Entity,
    "phone_number": Entity,
    "confirmed": BooleanEntity,
}

ner_chain = NERChain(
    llm=cohere_ner_llm,
    verbose=True,
    entities=entities,
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
