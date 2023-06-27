import json
from typing import List, Optional
from lib.conversation_memory import ConversationMemory
from lib.process.validation_chain import FormValidationChain
from pydantic import BaseModel, root_validator, validator
from lib.process.schemas import Process
import sys


class MyProcess(Process):
    first_name: Optional[str] = None

    @root_validator()
    def validate_first_name(cls, values: dict) -> dict:
        if values.get("first_name") == "error":   
            values["errors"]["first_name"] = "Some error"
        return values


def test_validation_chain_errors():
    process = MyProcess.parse_obj({"first_name": "error"})
    assert process.first_name is "error"
    assert process.errors == {"first_name": "Some error"}
    assert process.dict().get("errors") == {"first_name": "Some error"}


def test_validation_chain_validate_output():
    chain = FormValidationChain(
        input_variables=["entities"],
        output_variables=["variables", "result"],
        process=MyProcess,
        memory=ConversationMemory(),
    )
    entities = json.dumps([{"name": "first_name", "value": "error"}])
    output = chain.validate(inputs={"entities": entities})
    assert output["variables"]["errors"] == {"first_name": "Some error"}
