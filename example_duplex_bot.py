import datetime
import random
from typing import ClassVar, Optional

import yaml
from langchain.chat_models import ChatOpenAI
from pydantic import Field, root_validator, validator

from lib.bot import console_bot, gradio_bot
from lib.conversation_memory import ConversationMemory
from lib.logger_config import setup_logger
from lib.ner.entities.basic_entities import BooleanEntity, Entity, EntityExample
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.process.process_chain import ProcessChain
from lib.process.schemas import Process

logger = setup_logger(__name__)
# Nathan desperately needs a hair cut but he is too lazy to call himself.
# Fortunately, he has an AI assistant that can book an appointment for him.

# Let's define some availability for Nathan
now = datetime.datetime.now()


class DuplexProcess(Process):
    nathan_available_slots: ClassVar[list[datetime.datetime]] = [
        # Tomorrow 9am
        datetime.datetime(now.year, now.month, now.day + 1, 9, 0),
        # Tomorrow 2pm
        datetime.datetime(now.year, now.month, now.day + 1, 14, 0),
        # Tomorrow 6pm
        datetime.datetime(now.year, now.month, now.day + 1, 18, 0),
        # 2 days later 1pm
        datetime.datetime(now.year, now.month, now.day + 2, 13, 0),
        # 12pm in 3 days,
        datetime.datetime(now.year, now.month, now.day + 3, 12, 0),
        # 5pm in 3 days,
        datetime.datetime(now.year, now.month, now.day + 3, 17, 0),
        # 7am in 4 days,
        datetime.datetime(now.year, now.month, now.day + 4, 7, 0),
        # 3pm in 4 days,
        datetime.datetime(now.year, now.month, now.day + 4, 15, 0),
    ]

    process_description = f"""
You are Nathan's AI assistant. Nathan needs a haircut and you need to book an appointment with the salon.
The salon is the Human.
If the Human asks, you can share the following information about Nathan:
- Nathan's first name
- Requested service is a man haircut
- Nathan's last name is Zylbersztejn
- Nathan's phone number is 555-555-5555
- Nathan's hair color is chestnut

Only share if asked.
If the salon asks anything else about Nathan, say it's not relevant.
"""

    availability: Optional[dict | str | None] = Field(
        title="Availability for a haircut",
        # This is not Python interpolation, but rather a hint for the model that can propose
        # relevant dates in context
        question="Would you have some availability on {{matching_slots_in_human_friendly_format}}?",
        description=f"Providing availability",
    )

    confirmation: Optional[bool] = Field(
        title="Confirmation",
        question="Can we confirm the appointment on {{appointment_time}}?",
        name="Confirmation",
        description="We need a confirmation to make sure the salon has booked the appointment.",
    )

    appointment_time: Optional[str] = Field(
        title="Appointment in human frendly format",
    )

    # This variable will be set in the validation step, but will not be asked to the user, hence no `question``.
    appointment: Optional[dict[str, str]] = Field(
        title="Appointment slot ISO datetimes",
    )

    matching_slots: Optional[list[datetime.datetime]] = Field(
        title="Matching slots",
    )

    matching_slots_in_human_friendly_format: Optional[str] = Field(
        title="Matching slots in human frendly format",
    )

    @root_validator(pre=True)
    def validate(cls, values: dict):
        # Let's define a default value in case no availability is provided yet or
        # there is no matching slot between the Nathan's availability and the salon's availability
        values[
            "matching_slots_in_human_friendly_format"
        ] = cls.slots_in_human_friendly_format(
            random.sample(cls.nathan_available_slots, 3)
        )
        if values.get("availability") is not None:
            matching_slots = cls.get_matching_slots(values["availability"])
            if len(matching_slots) == 0:
                del values["availability"]
            elif len(matching_slots) == 1:
                logger.debug(f"Found a slot at {matching_slots[0]}")
                values["appointment"] = {
                    "start": matching_slots[0].isoformat(),
                    "end": (
                        matching_slots[0] + datetime.timedelta(minutes=15)
                    ).isoformat(),
                }
                values["availability"] = {
                    "start": values["appointment"]["start"],
                    "end": values["appointment"]["end"],
                    "grain": 15 * 60,
                }
                values["appointment_time"] = matching_slots[0].strftime(
                    "%A, %d %B %Y, %H:%M"
                )
                values["matching_slots"] = [matching_slots[0]]
                values[
                    "matching_slots_in_human_friendly_format"
                ] = cls.slots_in_human_friendly_format([matching_slots[0]])
            elif len(matching_slots) > 1:
                logger.debug(f"Found several slots: {matching_slots}")
                del values["availability"]
                if "appointment" in values:
                    del values["appointment"]
                values[
                    "matching_slots_in_human_friendly_format"
                ] = cls.slots_in_human_friendly_format(matching_slots)

        logger.debug("validated entity values:", values)
        return values

    @validator("confirmation", pre=True)
    def validate_confirmation(cls, v):
        assert v is True, "What information do you need to confirm?"
        return v

    @classmethod
    def get_matching_slots(cls, availability: dict):
        start = datetime.datetime.fromisoformat(availability["start"])
        end = datetime.datetime.fromisoformat(availability["end"])
        available_slots = []

        for slot in cls.nathan_available_slots:
            if start <= slot < end:
                available_slots.append(slot)

        return available_slots

    @classmethod
    def slots_in_human_friendly_format(cls, slots: list[datetime.datetime]):
        human_friendly_dates = [
            datetime.datetime.strftime(d, "%A %d at %H:%M") for d in slots
        ]
        return ", ".join(human_friendly_dates[:-1]) + " or " + human_friendly_dates[-1]

    def is_completed(self):
        return self.confirmation is True


ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-4")

process_chain = ProcessChain(
    memory=ConversationMemory(),
    ner_llm=ner_llm,
    chat_llm=chat_llm,
    process=DuplexProcess,
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

gradio_bot(
    chain=process_chain,
    initial_input="Hey",
    title="Duplex Bot",
).launch()
