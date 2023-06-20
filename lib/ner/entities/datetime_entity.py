import datetime
import json
from langchain import PromptTemplate
from pydantic import BaseModel, root_validator, validator


from langchain.chat_models import ChatOpenAI
from langchain.base_language import BaseLanguageModel
from langchain import LLMChain
from jinja2 import Template

from .basic_entities import Entity


class DateTime(BaseModel):
    start: str
    end: str
    grain: int

    @root_validator
    def validate(cls, values):
        try:
            start = datetime.datetime.fromisoformat(values["start"])
            end = datetime.datetime.fromisoformat(values["end"])
            if end < start:
                raise ValueError("End date is before start date")
            return values
        except ValueError as e:
            raise ValueError("Could not parse date time")


class DateTimeEntity(Entity):
    name: str = "datetime"
    value: str

    @validator("value")
    def validate_date(cls, v, values):
        chain = LLMChain(
            llm=values["llm"], prompt=DateTimeEntity.get_prompt(), verbose=True
        )
        result = chain.run({"query": v})
        value: str | None = None
        try:
            return DateTime.parse_obj(json.loads(result))
        except ValueError as e:
            value = None
            pass
        return value

    @staticmethod
    def get_prompt() -> PromptTemplate:
        import datetime

        current_time = datetime.datetime.now()

        # In a couple of hours
        in_couple_of_hours = current_time + datetime.timedelta(hours=2)

        # Tomorrow
        tomorrow = (current_time + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Thursday 2pm
        today = current_time.date()
        thursday = today + datetime.timedelta((3 - today.weekday()) % 7)
        thursday_2pm = datetime.datetime.combine(thursday, datetime.time(14, 0))

        # Next Tuesday 4pm
        next_tuesday = today + datetime.timedelta((7 - today.weekday() + 1) % 7)
        next_tuesday_4pm = datetime.datetime.combine(next_tuesday, datetime.time(16, 0))

        # Next week
        next_week = (current_time + datetime.timedelta(weeks=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        next_week = next_week - datetime.timedelta(days=next_week.weekday())

        # Next month
        next_month = (
            current_time.replace(day=1) + datetime.timedelta(days=32)
        ).replace(day=1)
        next_month = next_month.replace(hour=0, minute=0, second=0, microsecond=0)
        examples = []

        # In a couple of hours
        start_in_couple_of_hours = in_couple_of_hours.isoformat()
        end_in_couple_of_hours = (
            in_couple_of_hours + datetime.timedelta(hours=1, seconds=-1)
        ).isoformat()
        examples.append(
            {
                "text": "In a couple of hours",
                "result": {
                    "start": start_in_couple_of_hours,
                    "end": end_in_couple_of_hours,
                    "grain": 3600,
                },
            }
        )

        # Tomorrow
        start_tomorrow = tomorrow.isoformat()
        end_tomorrow = (tomorrow + datetime.timedelta(days=1, seconds=-1)).isoformat()
        examples.append(
            {
                "text": "Tomorrow",
                "result": {
                    "start": start_tomorrow,
                    "end": end_tomorrow,
                    "grain": 3600,
                },
            }
        )

        # Thursday 2pm
        start_thursday_2pm = thursday_2pm.isoformat()
        end_thursday_2pm = (
            thursday_2pm + datetime.timedelta(hours=1, seconds=-1)
        ).isoformat()
        examples.append(
            {
                "text": "Thursday 2pm",
                "result": {
                    "start": start_thursday_2pm,
                    "end": end_thursday_2pm,
                    "grain": 3600,
                },
            }
        )

        # Next Tuesday 4pm
        start_next_tuesday_4pm = next_tuesday_4pm.isoformat()
        end_next_tuesday_4pm = (
            next_tuesday_4pm + datetime.timedelta(hours=1, seconds=-1)
        ).isoformat()
        examples.append(
            {
                "text": "Next Tuesday 4pm",
                "result": {
                    "start": start_next_tuesday_4pm,
                    "end": end_next_tuesday_4pm,
                    "grain": 3600,
                },
            }
        )

        # Next week
        start_next_week = next_week.isoformat()
        end_next_week = (next_week + datetime.timedelta(days=7, seconds=-1)).isoformat()
        examples.append(
            {
                "text": "Next week",
                "result": {
                    "start": start_next_week,
                    "end": end_next_week,
                    "grain": 604800,
                },
            }
        )
        # Next month
        start_next_month = next_month.isoformat()
        one_month_duration = (
            next_month.replace(day=1) + datetime.timedelta(days=32)
        ).replace(day=1) - next_month
        end_next_month = (next_month + one_month_duration).isoformat()
        examples.append(
            {
                "text": "Next month",
                "result": {
                    "start": start_next_month,
                    "end": end_next_month,
                    "grain": one_month_duration.total_seconds(),
                },
            }
        )

        def escape_json(obj: dict) -> str:
            return json.dumps(obj)

        now = datetime.datetime.now()
        jinja_template = Template(
            """
At this very moment, date time is {{now}}.
For a given date expressed in natural language, output date time information in ISO format as shown in the examples:

EXAMPLES:
{% for e in examples %}
Natural language date:
{{e.text}}
result:
{{ escape_json(e.result) }}
{% endfor %}
END OF EXAMPLES

Natural language date:{% raw %}
{{query}}{% endraw %}
ISO:
"""
        )
        try:
            template: str = jinja_template.render(
                {
                    "examples": examples,
                    "now": now.strftime(
                        "%A, %B %d, %Y",
                    ),
                    "escape_json": escape_json,
                }
            )
            try:
                return PromptTemplate(
                    input_variables=[
                        "query",
                    ],
                    template_format="jinja2",
                    template=template,
                )
            except Exception as e:
                print(e)
                raise e

        except Exception as e:
            print(e)
            raise e


def main():
    from lib.ner.ner_chain import NERChain
    from .basic_entities import EntityExample

    examples = [
        {
            "text": "let's go next Thursday at 4pm",
            "entities": [{"name": "datetime", "value": "next Thursday at 4pm"}],
        },
        {
            "text": "What do you have next week?",
            "entities": [{"name": "datetime", "value": "next week"}],
        },
        {
            "context": "I can propose Wednesday 14 at 09:00, Friday 16 at 11:00 or Sunday 18 at 11:00. Is there an option that suits you? What is your availability like next week?",
            "text": "the second option is great",
            "entities": [{"name": "datetime", "value": "Friday 16 at 11:00"}],
        },
        {
            "context": "Would you be available Thursday 08 at 12:00 or Sunday 11 at 8:00?",
            "text": "let's go with the last one",
            "entities": [{"name": "datetime", "value": "Sunday 11 at 8:00"}],
        },
    ]

    openai_entity_extractor_llm = ChatOpenAI(
        temperature=0, client=None, max_tokens=256, model="gpt-3.5-turbo"
    )

    ner_chain = NERChain(
        llm=openai_entity_extractor_llm,
        verbose=True,
        entities={"datetime": DateTimeEntity},
        examples=[EntityExample.parse_obj(e) for e in examples],
    )

    from rich.console import Console
    from rich.prompt import Prompt
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


if __name__ == "__main__":
    main()
