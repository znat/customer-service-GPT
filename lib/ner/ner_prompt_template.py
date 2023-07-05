import json
from typing import Any, Optional, Type
from langchain.prompts.base import StringPromptTemplate
from pydantic import BaseModel
from .entities.basic_entities import EntityExample

PROMPT_FEW_SHOTS = """
Extract entities {variable_names} from the 'text', considering the 'context' as in the following examples.

{additional_instructions}

If no entities are found, just output [].

EXAMPLES:

{examples}

END OF EXAMPLES:

context: {{context}}
text: {{input}}
entities:"""

PROMPT_FINE_TUNED = """context: {{context}}
text: {{input}}
"""

class NERPromptTemplate(StringPromptTemplate):
    input_variables: list[str] = ["input"]
    additional_instructions: Optional[str] = ""
    debug: bool = False
    template: str = None
    examples: Optional[list[EntityExample]] = None
    entities: dict[str, Type[BaseModel]]

    def format(self, **kwargs: Any) -> str:
        variable_names = ", ".join(self.entities.keys())
        examples = self.stringify_dict_for_template(
            [
                example.dict(exclude={"entities": {"__all__": {"description"}}})
                for example in self.examples
            ]
        ) if self.examples else ""
        prompt_template = self.template
        if self.template is None:
            prompt_template = PROMPT_FEW_SHOTS if self.examples else PROMPT_FINE_TUNED

        template = prompt_template.format(
            variable_names=variable_names,
            examples=examples,
            additional_instructions=self.additional_instructions,
        )
        context = self.get_entity_extraction_context(kwargs["history"])
        return template.format(
            **{
                **kwargs,
                "context": context,
            }
        )

    def stringify_dict_for_template(self, dictionary: list[dict[str, Any]]) -> str:
        result = []
        for index, item in enumerate(dictionary):
            if item.get("context"):
                result.append(f"context: {item['context']}")
            result.append(f"text: {item['text']}")
            result.append("entities:")
            entities_str = (
                json.dumps(item["entities"], indent=2)
                if self.debug
                else json.dumps(item["entities"])
            )
            entities_str = entities_str.replace("{", "{{").replace("}", "}}")
            result.append(entities_str)
            if index < len(dictionary) - 1:
                result.append("\n")
        str_result = "\n".join(result)
        return str_result

    @staticmethod
    def get_entity_extraction_context(text: str) -> str:
        response_parts = text.split('AI: ')
        last_ai_response = response_parts[-1].strip()
        response = last_ai_response if last_ai_response else ""
        return response
