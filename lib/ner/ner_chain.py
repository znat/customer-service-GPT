from typing import Optional, Type
from langchain.chains.base import Chain
from langchain.chains.sequential import SequentialChain
from langchain.chains.transform import TransformChain
from langchain.base_language import BaseLanguageModel
from pydantic import BaseModel, root_validator
from langchain import LLMChain
import json

from .ner_prompt_template import NERPromptTemplate
from .entities.basic_entities import EntityExample, Entity


class NERChain(SequentialChain):
    input_variables: list[str] = ["input", "history"]
    output_variables: list[str] = ["entities"]
    additional_instructions: Optional[str] = ""
    output_key: str = "entities"
    llm: BaseLanguageModel
    examples: Optional[list[EntityExample]] = None
    entities: dict[str, Type[Entity] | tuple[Type[Entity], BaseLanguageModel]]
    chains: list[Chain] = []

    @staticmethod
    def parse_entities(
        entities_definition: dict[str, Type[BaseModel]],
        raw_entities: str,
        llm: BaseLanguageModel,
        verbose: bool = False,
    ) -> str:
        validated_entities = []
        # Dumb models might predict more that just entities and repeat examples
        # but they do generally ok at predicting a valid json object with entities.
        # So we just split by new line and take the first line.
        raw_entities = raw_entities.strip().split("\n")[0]

        try:
            for raw_entity in json.loads(raw_entities):
                entity_definition = entities_definition.get(raw_entity["name"], None)
                if entity_definition is not None:
                    entity_type: Type[Entity]
                    if isinstance(entity_definition, tuple):
                        entity_type, llm = entity_definition  # type: ignore
                        raw_entity["llm"] = llm
                    else:
                        entity_type = entity_definition  # type: ignore
                        parsed_entity: dict | None = None
                        try:
                            # Entity can use the llm to parse the value
                            parsed_entity = entity_type.parse_obj(
                                {**raw_entity, "llm": llm}
                            ).dict(include={"name", "value"})
                        except:
                            parsed_entity = None
                        # An invalid entity will have a null value and we don't want to include it
                        if parsed_entity and parsed_entity["value"] is not None:
                            validated_entities.append(parsed_entity)

        except json.JSONDecodeError as e:
            pass
            # raise ValueError(f"Could not parse entities: {raw_entities}")
        if verbose:
            print(f"Validated entities: {validated_entities}")
        return json.dumps(validated_entities)

    @root_validator(pre=True)
    def validate_chains(cls, values: dict) -> dict:
        if "chains" in values:
            raise ValueError("Cannot specify chains in NERChain")

        ner_chain = LLMChain(
            llm=values["llm"],
            verbose=values["verbose"],
            output_key="raw_entities",
            prompt=NERPromptTemplate(
                input_variables=["input", "history"],
                examples=values.get("examples", None),
                entities=values["entities"],
                additional_instructions=values.get("additional_instructions", None)
            ),
        )

        def transform(raw_entities: dict[str, str]) -> dict[str, str]:
            return {
                "entities": NERChain.parse_entities(
                    values["entities"],
                    raw_entities["raw_entities"],
                    values["llm"],
                    values["verbose"],
                )
            }

        transform_chain = TransformChain(
            input_variables=["raw_entities"],
            output_variables=["entities"],
            verbose=values["verbose"],
            transform=transform,
        )

        values["chains"] = [ner_chain, transform_chain]

        return values
