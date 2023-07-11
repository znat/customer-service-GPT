import datetime
import os
import random
from typing import ClassVar, Optional

import yaml
from langchain.chat_models import ChatOpenAI
from pydantic import Field, ValidationError, root_validator, validator

from lib.bot import gradio_bot
from lib.conversation_memory import ConversationMemory
from lib.logger_config import setup_logger
from lib.ner.entities.basic_entities import BooleanEntity, Entity, EntityExample
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.process.process_chain import ProcessChain
from lib.process.schemas import Process

logger = setup_logger(__name__)
# Some calendar availability the bot can use
now = datetime.datetime.now()


class AppointmentBookingProcess(Process):
    """
    This process is a simple appointment booking process."""

    # Let's populate the salon calendar with some availability.
    # The bot will match users availability with the salon calendar.
    salon_available_slots: ClassVar[list[datetime.datetime]] = [
        now + datetime.timedelta(days=1, hours=9),
        now + datetime.timedelta(days=1, hours=14),
        now + datetime.timedelta(days=1, hours=18),
        now + datetime.timedelta(days=2, hours=13),
        now + datetime.timedelta(days=3, hours=12),
        now + datetime.timedelta(days=3, hours=17),
        now + datetime.timedelta(days=4, hours=7),
        now + datetime.timedelta(days=4, hours=15),
        now + datetime.timedelta(days=6, hours=9),
        now + datetime.timedelta(days=5, hours=13),
        now + datetime.timedelta(days=5, hours=18),
        now + datetime.timedelta(days=7, hours=16),
        now + datetime.timedelta(days=7, hours=9),
        now + datetime.timedelta(days=8, hours=11),
        now + datetime.timedelta(days=9, hours=8),
        now + datetime.timedelta(days=9, hours=19),
        now + datetime.timedelta(days=10, hours=2),
    ]

    # The process description will be injected in the prompt template.
    process_description = f"""
You are a hair salon AI attendant and you are happy to help the User book an appointment.

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
        aknowledgement="Great, we have booked an appointment on {{human_friendly_appointment_time}}",
    )
    # This variable will be set in the validation step, but will not be asked to the user, hence no `question``.
    appointment: Optional[dict[str, str]] = Field(
        title="Appointment slot ISO datetimes",
    )

    first_name: Optional[str] = Field(
        title="First name",
        description="First name of the user, required to identify a patient",
        question="What is your first name?",
    )

    last_name: Optional[str] = Field(
        title="Last name",
        description="Last name of the user, required to identify a patient",
        question="What is your last name?",
    )

    phone_number: Optional[str] = Field(
        title="Phone number",
        description="Phone number of the user, required to contact a patient in case of unexpected events, such as delays or cancellations",
        question="What is your phone number?",
    )

    confirmation: Optional[bool] = Field(
        title="Confirmation",
        question="Great. Let's review everyting. You are {{first_name}} {{last_name}}, your phone number is {{phone_number}} and we have booked an appointment on {{human_friendly_appointment_time}}. Is everything correct?",
        name="Confirmation",
        description="We need a confirmation to make sure we have the right information before we book.",
    )

    matching_slots_in_human_friendly_format: Optional[str] = Field(
        title="Matching slots in human frendly format",
    )

    errors_count: Optional[int] = 0

    @classmethod
    def get_matching_slots(cls, availability: dict):
        start = datetime.datetime.fromisoformat(availability["start"])
        end = datetime.datetime.fromisoformat(availability["end"])
        available_slots = []

        for slot in cls.salon_available_slots:
            if start <= slot < end:
                available_slots.append(slot)

        return available_slots

    @classmethod
    def slots_in_human_friendly_format(cls, slots: list[datetime.datetime]):
        if len(slots) == 0:
            return ""
        elif len(slots) == 1:
            return datetime.datetime.strftime(slots[0], "%A %d at %H:%M")
        else:
            human_friendly_dates = [
                datetime.datetime.strftime(d, "%A %d at %H:%M") for d in slots
            ]
            return (
                ", ".join(human_friendly_dates[:-1]) + " or " + human_friendly_dates[-1]
            )

    def is_completed(self):
        return super().is_completed() and self.confirmation is True

    def is_failed(self):
        return self.errors_count and self.errors_count >= 3

    @root_validator(pre=True)
    def validate(cls, values: dict):
        values["errors"] = {}
        # Let's define a default value in case no availability is provided yet or
        # there is no matching slot between the Nathan's availability and the salon's availability
        values[
            "matching_slots_in_human_friendly_format"
        ] = cls.slots_in_human_friendly_format(
            random.sample(cls.salon_available_slots, 3)
        )
        if values.get("availability") is not None:
            # The grain of a datetime can be just one minute. We want to make sure we have at least 15 minutes
            if values["availability"]["grain"]  < 15 * 60:
                values["availability"] ={
                    "start": values["availability"]["start"],
                    "end": (datetime.datetime.fromisoformat(values["availability"]["start"]) + datetime.timedelta(minutes=15)).isoformat(),
                    "grain":  15 * 60,
                }
            matching_slots = cls.get_matching_slots(values["availability"])

            if len(matching_slots) == 0:
                logger.debug("No matching slot found")
                del values["availability"]
                values[
                    "matching_slots_in_human_friendly_format"
                ] = cls.slots_in_human_friendly_format(
                    random.sample(cls.salon_available_slots, 3)
                )
                values["errors"][
                    "availability"
                ] = "No, unfortunately. but we can offer {{matching_slots_in_human_friendly_format}}"

            elif len(matching_slots) == 1:
                logger.debug(f"Found a slot at {matching_slots[0]}")
                if values["availability"]["grain"] > 60 * 60:
                    del values["availability"]
                    values[
                        "matching_slots_in_human_friendly_format"
                    ] = cls.slots_in_human_friendly_format([matching_slots[0]])
                    values["errors"][
                        "availability"
                    ] = "We can propose you a slot on {{matching_slots_in_human_friendly_format}}. Would that work?"
                else:
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
                    values[
                        "matching_slots_in_human_friendly_format"
                    ] = cls.slots_in_human_friendly_format([matching_slots[0]])
            elif len(matching_slots) > 1:
                logger.debug(f"Found several slots: {matching_slots}")
                del values["availability"]
                values[
                    "matching_slots_in_human_friendly_format"
                ] = cls.slots_in_human_friendly_format(matching_slots)
                values["errors"][
                    "availability"
                ] = "We have several slot available: {{matching_slots_in_human_friendly_format}}. Would that work?"
        if values.get("phone_number") is not None:
            phone_number = values["phone_number"]
            import re

            pattern = re.compile(r"^\d{3}-\d{3}-\d{4}$")
            if not pattern.match(phone_number):
                del values["phone_number"]
                values["errors"][
                    "phone_number"
                ] = "Can you please provide a valid phone number using the XXX-XXX-XXXX format?"
                values["errors_count"] = values.get("errors_count", 0) + 1

        logger.debug(f"validated process values: {values}")
        return values

    @validator("first_name")
    def validate_first_name(cls, v):
        assert v is None or v[0].isalpha(), "First name must start with a letter."
        return v.capitalize()

    @validator("last_name")
    def validate_last_name(cls, v):
        assert v is None or v[0].isalpha(), "Last name must start with a letter."
        return v.capitalize()

    @validator("confirmation", pre=True)
    def validate_confirmation(cls, v):
        assert v is True, "What would you like to change?"
        return v


ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=200, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=200, model="gpt-3.5-turbo")

from langchain.llms import Cohere

# chat_cohere = Cohere(temperature=0, client=None)

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
        for e in yaml.safe_load(
            open(
                os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    "booking_bot_entity_examples.yaml",
                )
            )
        )
    ],
)


gradio_bot(
    chain=process_chain,
    title="Appointment Booking",
    initial_input="Hey",
).launch()
