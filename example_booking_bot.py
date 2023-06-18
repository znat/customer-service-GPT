import datetime
import random
from typing import ClassVar, Optional

import yaml
from langchain.chat_models import ChatOpenAI
from pydantic import Field, root_validator, validator

from lib.bot import gradio_bot
from lib.conversation_memory import ConversationMemory
from lib.logger_config import setup_logger
from lib.ner.entities.basic_entities import (BooleanEntity, Entity,
                                             EntityExample)
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.process.process_chain import ProcessChain
from lib.process.schemas import Process

logger = setup_logger(__name__)


# Some calendar availability the bot can use
now = datetime.datetime.now()

# Let's populate the salon calendar with some availability.
# The bot will match users availability with the salon calendar.


# Use the join() function to create a string with the bulleted list of dates


class AppointmentBookingProcess(Process):
    salon_available_slots: ClassVar[list[datetime.datetime]] = [
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
        # 9am in 6 days,
        datetime.datetime(now.year, now.month, now.day + 6, 9, 0),
        # 1pm in 5 days,
        datetime.datetime(now.year, now.month, now.day + 5, 13, 0),
        # 6pm in 5 days,
        datetime.datetime(now.year, now.month, now.day + 5, 18, 0),
        # 4pm in 7 days,
        datetime.datetime(now.year, now.month, now.day + 7, 16, 0),
        # 9am in 7 days,
        datetime.datetime(now.year, now.month, now.day + 7, 9, 0),
        # 11am in 8 days,
        datetime.datetime(now.year, now.month, now.day + 8, 11, 0),
        # 8am in 9 days,
        datetime.datetime(now.year, now.month, now.day + 9, 8, 0),
        # 7pm in 9 days,
        datetime.datetime(now.year, now.month, now.day + 9, 19, 0),
        # 2am in 10 days,
        datetime.datetime(now.year, now.month, now.day + 10, 2, 0),
    ]

    process_description = f"""
You are a hair salon attendant AI. The User wants to book and appointment

You can share the following information with the User if the User asks:
- Address of the salon is 123 Main Street, CoolVille, QC, H3Z 2Y7
- The salon phone number is 514-666-7777

To all other questions reply you don't know. 

"""

    availability: Optional[dict | str] = Field(
        title="Availability for doctor appointment",
        question="Would you be available {{matching_slots_in_human_friendly_format}}",
        description=f"Providing availability helps finding an available slot in the salon's calendar",
    )

    appointment_time: Optional[str] = Field(
        title="Appointment in human frendly format",
    )
    # This variable will be set in the validation step, but will not be asked to the user, hence no `question``.
    appointment: Optional[dict[str, str]] = Field(
        title="Appointment slot ISO datetimes",
    )

    first_name: Optional[str] = Field(
        name="First name",
        description="First name of the user, required to identify a patient",
        question="What is your first name?",
    )

    last_name: Optional[str] = Field(
        name="Last name",
        description="Last name of the user, required to identify a patient",
        question="What is your last name?",
    )

    phone_number: Optional[str] = Field(
        name="Phone number",
        description="Phone number of the user, required to contact a patient in case of unexpected events, such as delays or cancellations",
        question="What is your phone number?",
    )

    confirmation: Optional[bool] = Field(
        title="Confirmation",
        question="Great. Let's review everyting. You are {{first_name}} {{last_name}}, your phone number is {{phone_number}} and we have booked an appointment on {human_friendly_appointment_time}. Is everything correct?",
        name="Confirmation",
        description="We need a confirmation to make sure we have the right information before we book.",
    )

    matching_slots: Optional[list[datetime.datetime]] = Field(
        title="Matching slots",
    )

    matching_slots_in_human_friendly_format: Optional[str] = Field(
        title="Matching slots in human frendly format",
    )

    @classmethod
    def matching_slots(cls, availability: dict):
        start = datetime.datetime.fromisoformat(availability["start"])
        end = datetime.datetime.fromisoformat(availability["end"])
        available_slots = []

        for slot in cls.salon_available_slots:
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

    @root_validator(pre=True)
    def validate(cls, values: dict):
        # Let's define a default value in case no availability is provided yet or
        # there is no matching slot between the user availability and the salon availability
        values[
            "matching_slots_in_human_friendly_format"
        ] = cls.slots_in_human_friendly_format(
            random.sample(cls.salon_available_slots, 3)
        )
        if values.get("availability") is not None:
            matching_slots = cls.matching_slots(values["availability"])
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
                    "grain": 15*60,
                }
                values["appointment_time"] = matching_slots[0].strftime(
                    "%A, %d %B %Y, %H:%M"
                )
                values["matching_slots"] = [matching_slots[0]]
                values[
                    "matching_slots_in_human_friendly_format"
                ] = cls.slots_in_human_friendly_format([matching_slots[0]])
            elif len(matching_slots) > 1:
                logger.debug(f"Found several slots")
                del values["availability"]
                if "appointment" in values:
                    del values["appointment"]
                values[
                    "matching_slots_in_human_friendly_format"
                ] = cls.slots_in_human_friendly_format(matching_slots)
       
        logger.debug("validated entity values:", values)
        return values

    @validator("first_name")
    def validate_first_name(cls, v):
        assert v is None or v[0].isalpha(), "First name must start with a letter."
        return v.capitalize()

    @validator("last_name")
    def validate_last_name(cls, v):
        assert v is None or v[0].isalpha(), "Last name must start with a letter."
        return v.capitalize()

    @validator("phone_number")
    def validate_phone_number(cls, v):
        import re

        pattern = re.compile(r"^\d{3}-\d{3}-\d{4}$")
        assert v is None or pattern.match(
            v
        ), f"Can you please provide a valid phone number using the XXX-XXX-XXXX format?"
        return v

    @validator("confirmation", pre=True)
    def validate_confirmation(cls, v):
        assert v is True, "What would you like to change?"
        return v


ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")

process_chain = ProcessChain(
    memory=ConversationMemory(),
    ner_llm=ner_llm,
    chat_llm=chat_llm,
    process=AppointmentBookingProcess,
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

gradio_bot(
    chain=process_chain,
    title="Appointment Booking",
    initial_input="I want to book an appointment",
).launch()
