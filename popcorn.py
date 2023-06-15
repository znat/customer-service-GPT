import yaml
from example_booking_bot import AppointmentBookingProcess
from example_duplex_bot import DuplexProcess
from lib.conversation_memory import ConversationMemory
from lib.ner.entities.basic_entities import BooleanEntity, Entity, EntityExample
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.process.process_chain import ProcessChain

from langchain.chat_models import ChatOpenAI

ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-4")

customer_chain = ProcessChain(
    memory=ConversationMemory(),
    ner_llm=ner_llm,
    chat_llm=chat_llm,
    form=AppointmentBookingProcess,
    verbose=True,
    entities={
        "availability": DateTimeEntity,
        "first_name": Entity,
        "last_name": Entity,
        "phone_number": Entity,
        "confirmation": BooleanEntity,
    },  # type: ignore
    additional_ner_instructions="""
- "confirmation" entity: should only be `true` if the gives an explicity confirmation that all the collected information is correct
and not if the user just says "yes" or confirms but asks follow-up questions.
- "phone number" entity: should be in the XXX-XXX-XXXX format. Format if and only if the user provides a phone number with 10 digits.
    """,
    entity_examples=[
        EntityExample.parse_obj(e)
        for e in yaml.safe_load(open("booking_bot_entity_examples.yaml"))
    ],
)

salon_chain = ProcessChain(
    memory=ConversationMemory(),
    ner_llm=ner_llm,
    chat_llm=chat_llm,
    form=DuplexProcess,
    verbose=True,
    entities={
        "availability": DateTimeEntity,
        "confirmation": BooleanEntity,
    },
    entity_examples=[
        EntityExample.parse_obj(e)
        for e in yaml.safe_load(open("duplex_bot_entity_examples.yaml"))
    ],
)

gradio_bot(chain=process_chain, initial_input="Hey", title="Duplex Bot").launch()