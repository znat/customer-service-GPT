from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Type
import json
from pydantic import BaseModel, ValidationError

from .schemas import Result, Status, Process

from ..conversation_memory import ConversationMemory
from ..logger_config import setup_logger
from .. import utils

logger = setup_logger(__name__)


class ProcessValidationChain(Chain):
    completed_variable: str = "_completed"
    process: Type[Process]
    input_variables: List[str]
    output_variables: List[str]
    memory: ConversationMemory

    @property
    def input_keys(self) -> List[str]:
        """Expect input keys.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return output keys.

        :meta private:
        """
        return self.output_variables

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return self.validate(inputs)

    def load_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        for field in self.process.__fields__.keys():
            if field not in variables.keys():
                stored_value = self.memory.kv_store.get(field)
                if stored_value is not None:
                    variables[field] = self.memory.kv_store.get(field)
        return variables

    def save_variables(self, variables: Dict[str, Any]) -> None:
        for field in self.process.__fields__.keys():
            if field in variables.keys():
                self.memory.kv_store.set(field, variables[field])
            else:
                self.memory.kv_store.set(field, None)

    def validate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Just to raise the "All fields should be optional" error if applicable
        self.process()  # type: ignore
        try:
            entities = json.loads(inputs["entities"])
        except json.JSONDecodeError as e:
            entities = []

        variables_from_entities = self.load_variables(
            {d["name"]: d["value"] for d in entities}
        )
        result: Result | None = None
        diff = []
        try:
            logger.debug(
                f"Current variables: {variables_from_entities}",
            )
            data = self.process.parse_obj(variables_from_entities)
            diff = utils.dict_diff(
                after=data.dict(),
                before=variables_from_entities,
            )
            self.save_variables(data.dict())
            logger.debug(
                f"Process model post-validation: {data.dict()}",
            )

            if data.is_completed():
                result = Result(
                    status=Status.completed, result=data, errors=data.errors
                )
            if data.is_failed():
                result = Result(status=Status.failed, result=data, errors=None)
            self.memory.kv_store.set("errors", {})
            variables_from_entities = self.memory.kv_store.load_memory_variables()[
                "variables"
            ]
            variables_from_entities["errors"] = data.errors
        except ValidationError as e:
            logger.debug(
                f"Validation error: {e}",
            )
            errors = self.convert_validation_error_to_dict(e, "assertion")
            variables_from_entities = {
                k: v
                for k, v in self.memory.kv_store.load_memory_variables()[
                    "variables"
                ].items()
                if k not in errors.keys()
            }
            logger.debug("Variables after validation errors:", variables_from_entities)
            variables_from_entities["errors"] = errors
        return {
            "variables": variables_from_entities,
            "diff": diff,
            "result": result.dict() if result else None,
        }

    def variables_diff(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> list[str]:
        result = []
        diff_list = utils.dict_diff(before, after)
        for diff in diff_list:
            result.append(f"- {diff['name']} = {diff['value']} ({diff['operation']})")
        return result

    @staticmethod
    def convert_validation_error_to_dict(
        error: ValidationError, error_type: str
    ) -> dict:
        error_dict = {}

        for error_item in error.errors():
            field_name = error_item["loc"][0]
            message = error_item["msg"]
            error_type_from_error = error_item["type"]

            if (
                error_type == "missing"
                and error_type_from_error == "value_error.missing"
            ):
                error_dict[field_name] = message
            elif (
                error_type == "assertion"
                and error_type_from_error != "value_error.missing"
            ):
                error_dict[field_name] = message

        return error_dict
