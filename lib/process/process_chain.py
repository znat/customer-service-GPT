from pydantic import root_validator
from .schemas import Process
from ..conversation_memory import ConversationMemory
from ..ner.entities.basic_entities import Entity, EntityExample
from .validation_chain import FormValidationChain
from .process_prompt_template import ProcessPromptTemplate
from ..ner.ner_chain import NERChain
from langchain.chains.sequential import SequentialChain
from typing import Any, Callable, Dict, List, Optional, Type
from langchain import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain


class ProcessChain(SequentialChain):
    ner_llm: BaseLanguageModel
    chat_llm: BaseLanguageModel
    entities: dict[str, Type[Entity] | tuple[Type[Entity], BaseLanguageModel]]
    entity_examples: list[EntityExample]
    additional_ner_instructions: Optional[str] = ""
    process: Type[Process]
    memory: Optional[ConversationMemory]
    chains: Optional[list[Chain]] = []
    verbose: bool = True
    input_variables: Optional[List[str]] = ["input"]
    output_variables: Optional[List[str]] = ["response", "result"]

    @root_validator(pre=True)
    def validate_chains(cls, values: dict) -> dict:
        if "chains" in values:
            raise ValueError("Cannot specify chains in ProcessChain")
        values["chains"] = [
            NERChain(
                llm=values["ner_llm"],
                entities=values["entities"],
                examples=values["entity_examples"],
                callbacks=values.get("callbacks"),
                additional_instructions=values["additional_ner_instructions"]
                if "additional_ner_instructions" in values
                else None,
                verbose=values["verbose"],
            ),
            FormValidationChain(
                input_variables=["entities"],
                output_variables=["variables", "result"],
                process=values["process"],
                memory=values["memory"],
                callbacks=values.get("callbacks"),
                verbose=values["verbose"],
            ),
            ConversationChain(
                llm=values["chat_llm"],
                verbose=values["verbose"],
                callbacks=values.get("callbacks"),
                prompt=ProcessPromptTemplate(
                    input_variables=["input", "history", "variables"],
                    process=values["process"],
                    validate_template=False,
                ),
                memory=values["memory"],
            ),
        ]
        return values

    def set_callbacks(self, callbacks: list[BaseCallbackHandler]) -> None:
        """Set callbacks for all chains."""
        self.callbacks = callbacks
        if self.chains:
            for chain in self.chains:
                chain.callbacks = callbacks

    def reset(self) -> None:
        """Set memory for all chains."""
        self.memory = ConversationMemory()
        if self.chains and len(self.chains) > 2:
            self.chains[1].memory = self.memory
            self.chains[2].memory = self.memory

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs.
        The difference with super is that it doesn't call save_context on the memory
        which is not necessary since it was called from the chains when needed
        """
        self._validate_outputs(outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}
