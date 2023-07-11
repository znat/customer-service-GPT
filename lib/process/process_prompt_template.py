import json
import os
from typing import Any, Tuple, Type

from jinja2 import Template
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel, root_validator

from lib.logger_config import setup_logger

logger = setup_logger(__name__)
from ..utils import convert_list_to_string
from .schemas import Process


class ProcessPromptTemplate(PromptTemplate):
    input_variables: list[str] = [
        "input",
        "history",
    ]
    template: str = open(
        os.path.join(os.path.dirname(__file__), "process_prompt_template.jinja2")
    ).read()

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

        errors: dict | None = kwargs["variables"].get("errors")

        error_message = (
            errors.get(list(errors.keys())[0]) if errors and len(errors) else None
        )
        if error_message is not None:
            error_message = Template(error_message).render(**kwargs["variables"])

        next_variable_to_collect = (
            list(remaining_dict.keys())[0] if len(remaining_dict.keys()) > 0 else None
        )
        next_variable_question = ""
        if next_variable_to_collect is not None:
            next_variable_question = Template(
                self.process.schema()["properties"][next_variable_to_collect][
                    "question"
                ]
            ).render(**kwargs["variables"])

        return Template(self.template, lstrip_blocks=True, trim_blocks=True).render(
            goal=self.process.process_description,
            is_process_starting=self.is_first_message(kwargs["history"]),
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
            updates=self.get_updates(kwargs["diff"], kwargs["variables"]),
            **kwargs,
        )

    def is_first_message(self, history: str) -> bool:
        return "AI:" not in history

    def get_collected_variables(self, variables: dict[str, Any]) -> dict[str, Any]:
        # Filter out None values and errors
        return {k: v for k, v in variables.items() if v is not None and k != "errors"}

    def get_collected_variable_names(self, variables: dict[str, Any]) -> list[str]:
        return list(self.get_collected_variables(variables).keys())

    def get_remaining_variables_to_collect(
        self, variables: dict[str, Any] = {}
    ) -> Tuple[dict[str, Any], str, str]:
        model_schema = self.process.schema()
        fields = model_schema["properties"]
        logger.debug(f"variables: {variables}")
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
            entry = f"{index}. {key} ({value['variable_name']}): {value['description']}"
            result.append(entry)

        return "\n".join(result)

    def get_updates(self, diff: list[dict], variables: dict) -> str:
        additions = []
        updates = []
        schema = self.process.schema()
        for item in diff:
            if (
                "question" in schema["properties"][item["name"]]
                or "aknowledgement" in schema["properties"][item["name"]]
            ):
                if item["operation"] == "added":
                    additions.append(f"{item['name']}")
                elif item["operation"] == "updated":
                    updates.append(f"{item['name']}")

        if not additions and not updates:
            return ""

        additions_str = ", ".join([f"`{a}`" for a in additions]) if additions else ""
        updates_str = ", ".join([f"`{u}`" for u in updates]) if updates else ""

        all_vars = additions + updates
        all_vars_with_values = [f"`{var}`: \"{variables[var]}\"" for var in all_vars]
        all_vars_str = ", ".join(all_vars_with_values) if all_vars_with_values else "No variables"
        output = ""
        if additions and updates:
            output = f"- User provided {additions_str} and updated {updates_str}. Aknowledge the values of {all_vars_str}."
        elif additions and not updates:
            output = f"- User provided {additions_str}. Aknowledge the values of {all_vars_str}."
        elif not additions and updates:
            output = f"- User updated {updates_str}. Aknowledge the values of {all_vars_str}."

        return output
