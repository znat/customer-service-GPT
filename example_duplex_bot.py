import datetime
from typing import Optional

import yaml
from langchain.chat_models import ChatOpenAI
from pydantic import Field, root_validator, validator

from lib.bot import console_bot, gradio_bot
from lib.conversation_memory import ConversationMemory
from lib.ner.entities.basic_entities import BooleanEntity, Entity, EntityExample
from lib.ner.entities.datetime_entity import DateTimeEntity
from lib.process.process_chain import ProcessChain
from lib.process.schemas import Process

# Nathan desperately needs a hair cut but he is too lazy to call himself.
# Fortunately, he has an AI assistant that can book an appointment for him.

# Let's define some availability for Nathan
now = datetime.datetime.now()
nathan_available_slots = [
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

# Use the join() function to create a string with the bulleted list of dates
bulleted_list = "\n".join(
    "- {}".format(iso_date)
    for iso_date in [slot.isoformat() for slot in nathan_available_slots]
)


class DuplexProcess(Process):
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

Nathan's has the following available slots:

{bulleted_list}
"""

    availability: Optional[dict | str | None] = Field(
        title="Availability for a haircut",
        # This is not Python interpolation, but rather a hint for the model that can propose
        # relevant dates in context
        question="Would you have some availability on {2 available slots formatted as '%A %B %-d %H:%M', e.g 'Monday June 3 15:00' format}?",
        description=f"Providing availability",
        exclude=True,  # We dont need it in the final result
    )

    appointment_time: Optional[str] = Field(
        title="Appointment in human frendly format",
    )
    # This variable will be set in the validation step, but will not be asked to the user, hence no `question``.
    appointment: Optional[dict[str, str]] = Field(
        title="Appointment slot ISO datetimes",
    )

    confirmation: Optional[bool] = Field(
        title="Confirmation",
        question=f"Can we confirm the appointment on {{appointment_time}}?",
        name="Confirmation",
        description="We need a confirmation to make sure the salong has booked the appointment.",
        exclude=True,  # We dont need it in the final result
    )

    @classmethod
    def get_available_slots(from_date: Optional[datetime.datetime], to_date: Optional[datetime.datetime]):
        slots = [
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

    @root_validator(pre=True)
    def validate(cls, values: dict):
        if values.get("availability") is not None:
            start = datetime.datetime.fromisoformat(values["availability"]["start"])
            end = datetime.datetime.fromisoformat(values["availability"]["end"])
            matching_slots = []
            for a in nathan_available_slots:
                # Check if a 15 min appointment fits into the availability range provided by the user
                if start <= a <= a + datetime.timedelta(minutes=15) <= end:
                    matching_slots.append(a)
            if len(matching_slots) == 0:
                del values["availability"]
                print(f"No slot found in {nathan_available_slots}")
                raise ValueError(
                    "Nathan is not available on that date. Other available slots are {2_available_slots_in_human_friendly_format})}"
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
                    "Nathan can do {human_friendly_available_slots_matching_salon_availability}. Any works?"
                )
        print(values)
        return values

    @validator("confirmation", pre=True)
    def validate_confirmation(cls, v):
        assert v is True, "What information do you need to confirm?"
        return v

    def is_completed(self):
        return self.confirmation is True


ner_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")
chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=100, model="gpt-3.5-turbo")
from langchain.llms import Cohere

# ner_llm = Cohere(temperature=0, client=None, max_tokens=100)
# chat_llm = Cohere(temperature=0, client=None, max_tokens=100)


process_chain = ProcessChain(
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

# console_bot(chain=process_chain, initial_input="Hey")
gradio_bot(chain=process_chain, initial_input="Hey", title="Duplex Bot").launch()
