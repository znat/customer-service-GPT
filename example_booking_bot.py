import datetime
from typing import Optional

import yaml
from langchain.chat_models import ChatOpenAI
from pydantic import Field, root_validator, validator

from lib.bot import gradio_bot
from lib.conversation_memory import ConversationMemory
from lib.ner.entities.basic_entities import BooleanEntity, Entity, EntityExample
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.process.process_chain import ProcessChain
from lib.process.schemas import Process


# Some calendar availability the bot can use
now = datetime.datetime.now()

# Let's populate the clinic calendar with some availability.
# The bot will match users availability with the clinic calendar.
clinic_available_slots = [
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


def to_human_friendly_dates(dates):
    """Convert a list of datetime objects to a human friendly string"""
    human_friendly_dates = [
        datetime.datetime.strftime(d, "%A %d at %H:%M") for d in dates
    ]
    return ", ".join(human_friendly_dates[:-1]) + " or " + human_friendly_dates[-1]


# Use the join() function to create a string with the bulleted list of dates
bulleted_list = "\n".join(
    "- {}".format(iso_date)
    for iso_date in [slot.isoformat() for slot in clinic_available_slots]
)


class AppointmentBookingProcess(Process):
    process_description = f"""
You are a medical secretary AI. The User wants to book and appointment and your goal is collect the necessary information to book an appointment. in one of the following available slots:

{bulleted_list}

You can share the following information with the User if the User asks:
- Address of the clinic is 123 Main Street, CoolVille, QC, H3Z 2Y7
- The clinic phone number is 514-666-7777
- What to bring: social insurance card

To all other questions reply you don't know. If it's a medical question, reply that should be asked to the doctor.

"""

    availability: Optional[dict | str | None] = Field(
        title="Availability for doctor appointment",
        question="Would you be available {2_available_slots_in_human_friendly_format}",
        description=f"Providing availability helps finding an available slot in the clinics calendar",
        exclude=True,  # We dont need it in the final result
    )

    appointment_time: str = Field(
        title="Appointment in human frendly format",
    )
    # This variable will be set in the validation step, but will not be asked to the user, hence no `question``.
    appointment: dict[str, str] = Field(
        title="Appointment slot ISO datetimes",
    )

    first_name: str = Field(
        name="First name",
        description="First name of the user, required to identify a patient",
        question="What is your first name?",
    )

    last_name: str = Field(
        name="Last name",
        description="Last name of the user, required to identify a patient",
        question="What is your last name?",
    )

    phone_number: str = Field(
        name="Phone number",
        description="Phone number of the user, required to contact a patient in case of unexpected events, such as delays or cancellations",
        question="What is your phone number?",
    )

    confirmation: Optional[bool] = Field(
        title="Confirmation",
        question="Great. Let's review everyting. You are {first_name} {last_name}, your phone numner is {phone_number} and we have booked an appointment on {human_friendly_appointment_time}. Is everything correct?",
        name="Confirmation",
        description="We need a confirmation to make sure we have the right information before we book.",
        exclude=True,  # We dont need it in the final result
    )

    def is_completed(self):
        return self.confirmation is True

    @root_validator(pre=True)
    def validate(cls, values: dict):
        if values.get("availability") is not None:
            start = datetime.datetime.fromisoformat(values["availability"]["start"])
            end = datetime.datetime.fromisoformat(values["availability"]["end"])
            matching_slots = []
            for a in clinic_available_slots:
                # Check if a 15 min appointment fits into the availability range provided by the user
                if start <= a <= a + datetime.timedelta(minutes=15) <= end:
                    matching_slots.append(a)
            if len(matching_slots) == 0:
                del values["availability"]
                print(f"No slot found in {clinic_available_slots}")
                raise ValueError(
                    "We have no availability on that date. Other available slots are {2_available_slots_in_human_friendly_format}"
                )
            elif len(matching_slots) == 1:
                print(f"Found a slot at {matching_slots[0]}")
                values["appointment"] = {
                    "start": matching_slots[0].isoformat(),
                    "end": (
                        matching_slots[0] + datetime.timedelta(minutes=15)
                    ).isoformat(),
                }
                values["availability"] = matching_slots[0].isoformat()
                values["appointment_time"] = matching_slots[0].strftime(
                    "%A, %d %B %Y, %H:%M"
                )
            elif len(matching_slots) > 1:
                print(f"Found several slots at {matching_slots}")
                del values["availability"]
                raise ValueError(
                    "I can propose {human_friendly_available_slots_matching_salon_availability}. Is there an option that suits you?"
                )
        print(values)
        return values

    @validator("first_name")
    def validate_first_name(cls, v):
        assert v[0].isalpha(), "First name must start with a letter."
        return v.capitalize()

    @validator("last_name")
    def validate_last_name(cls, v):
        assert v[0].isalpha(), "Last name must start with a letter."
        return v.capitalize()

    @validator("phone_number")
    def validate_phone_number(cls, v):
        import re

        pattern = re.compile(r"^\d{3}-\d{3}-\d{4}$")
        assert pattern.match(
            v
        ), f"Can you please provide a valid phone number using the XXX-XXX-XXXX format?"
        return v

    @validator("confirmation", pre=True)
    def validate_confirmation(cls, v):
        assert v is True, "What would you like to change?"
        return v


ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-4")

process_chain = ProcessChain(
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

gradio_bot(
    chain=process_chain, initial_input="Hey", title="Appointment Booking"
).launch()
