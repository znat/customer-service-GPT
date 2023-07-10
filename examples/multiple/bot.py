import os
from langchain import LLMMathChain

import yaml
from langchain.chat_models import ChatOpenAI

from .simple_form_process import SimpleForm
from lib.bot import gradio_bot
from lib.conversation_memory import ConversationMemory
from lib.ner.entities.basic_entities import (IntEntity, Entity,
                                             EntityExample)
from lib.process.process_chain import ProcessChain

ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=200, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=200, model="gpt-3.5-turbo")

llm_math = LLMMathChain.from_llm(chat_llm, verbose=True)

process_chain = ProcessChain(
    memory=ConversationMemory(),
    ner_llm=ner_llm,
    chat_llm=chat_llm,
    process=SimpleForm,
    verbose=True,
    entities={
        "first_name": Entity,
        "age": IntEntity,
    },  # type: ignore
   
    entity_examples=[
        EntityExample.parse_obj(e)
        for e in yaml.safe_load(
            open(
                os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    "ner_data.yaml",
                )
            )
        )
    ],
)


gradio_bot(
    chain=llm_math,
    title="Appointment Booking",
    initial_input="Hey",
).launch()