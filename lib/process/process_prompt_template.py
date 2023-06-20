import json
from typing import Any, Optional, Tuple, Type

from jinja2 import Template
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel, root_validator

from .schemas import Process

from ..utils import convert_list_to_string

DEFAULT_PROMPT = """START CONTEXT
{{goal}}

Follow these rules when conversing with the Human:
- You can only use the context and the knowledge you have collected from this conversation to answer the User.
- You cannot use your pre-existing knowledge of the outside world.
- When the User asks a question, answer with the context of this conversation only. If the answer is not in the context, say you don't know and repeat your question.
- If the user does not answer the question, reply and repeat your question.
- You must predict one and only one AI message.
- introduce yourself if it's your first message


{% if collected|length > 2 %}
You already know the following information from the Human:

```json
{{collected}}
```
{% endif %}
{% if remaining|length > 2 %}
You don't know {{remaining_as_string}} yet and you have to collect then in the right order:

{{remaining}}

{% endif %}
END CONTEXT

{% if errors|length > 2 %}
Provide the following feedback to the Human and ask them to correct their answer:
{{error_message}}
{% elif errors|length <= 2 and remaining|length > 2 %}
Ask the `{{next_variable_to_collect}}` of the Human using the "question to ask" defined above.
{% else %}
You have successfully completed your task. Congratulations!
{% endif %}


{{history}}
User: {{input}}
AI:"""


class ProcessPromptTemplate(PromptTemplate):
    input_variables: list[str] = [
        "input",
        "history",
    ]
    template: str = DEFAULT_PROMPT

    template_format: str = "jinja2"
    validate_template: bool = True
    process: Type[Process]

    def format(self, **kwargs: Any) -> str:
        collected = self.get_collected_variables(kwargs["variables"])
        (
            remaining_dict,
            remaining,
            remaining_as_list,
        ) = self.get_remaining_variables_to_collect(kwargs["variables"])
        errors: dict | None = kwargs["variables"].get("_errors")
        error_message = (
            errors.get(list(errors.keys())[0]) if errors and len(errors) else None
        )
        next_variable_to_collect = (
            list(remaining_dict.keys())[0] if len(remaining_dict.keys()) > 0 else None
        )
        next_variable_question = ""
        if next_variable_to_collect is not None:
            next_variable_question = Template(
                self.process.schema()["properties"][next_variable_to_collect]["question"]
            ).render(**kwargs["variables"])

        return super().format(
            goal=self.process.process_description,
            remaining=remaining_as_list,
            collected=json.dumps(collected, indent=2),
            error_message=error_message,
            all_as_string=convert_list_to_string(list(kwargs["variables"].keys())),
            remaining_as_string=convert_list_to_string(
                [key for key in remaining_dict.keys() if not key.startswith("_")]
            ),
            next_variable_to_collect=next_variable_to_collect,
            next_variable_question=next_variable_question,
            errors=json.dumps(errors, indent=2) if errors is not None else None,
            **kwargs,
        )

    def get_collected_variables(self, variables: dict[str, Any]) -> dict[str, Any]:
        # Filter out None values and errors
        return {
            k: v for k, v in variables.items() if v is not None and k is not "_errors"
        }

    def get_collected_variable_names(self, variables: dict[str, Any]) -> list[str]:
        return list(self.get_collected_variables(variables).keys())

    def get_remaining_variables_to_collect(
        self, variables: dict[str, Any] = {}
    ) -> Tuple[dict[str, Any], str, str]:
        model_schema = self.process.schema()
        fields = model_schema["properties"]
        print("variables::",variables)
        json_object = {}
        for field_name, field_info in fields.items():
            if field_name not in self.get_collected_variables(
                variables
            ).keys() and field_info.get("question"):
                json_object[field_name] = {
                    "description": field_info.get("description", ""),
                    "variable_name": field_info.get("title", ""),
                }
                if field_info.get("question"):
                    json_object[field_name]["question"] = Template(
                        field_info["question"]
                    ).render(variables)
        return (
            json_object,
            json.dumps(json_object, indent=2),
            self.convert_dict_to_ordered_list(json_object),
        )

    def convert_dict_to_ordered_list(self, data_dict: dict):
        result = []
        for index, (key, value) in enumerate(data_dict.items(), start=1):
            entry = f"{index}. {key}\n"
            entry += f"    question to ask: {value['question']}\n"
            entry += f"    description: {value['description']}\n"
            result.append(entry)

        return "\n".join(result)
