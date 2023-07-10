import json
from typing import List, Optional

import pytest
from lib.conversation_memory import ConversationMemory
from lib.process.validation_chain import ProcessValidationChain
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
    chain = ProcessValidationChain(
        input_variables=["entities"],
        output_variables=["variables", "result"],
        process=MyProcess,
        memory=ConversationMemory(),
    )
    entities = json.dumps([{"name": "first_name", "value": "error"}])
    output = chain.validate(inputs={"entities": entities})
    assert output["variables"]["errors"] == {"first_name": "Some error"}

@pytest.mark.parametrize(
    "after, before, expected",
    [
        (
           {}, 
           {}, 
           []
        ),
        (
            {'a': 1, 'b': 2}, 
            {'a': 1, 'b': 20, 'c': 30},
            [
                "- b = 20 (changed)",
                "- c = 30 (added)"
            ] 
        ),
        (
            {'a': 1, 'b': 2, 'c': 3}, 
            {},
            [
                "- a = 1 (deleted)",
                "- b = 2 (deleted)",
                "- c = 3 (deleted)"
            ]
        ),
        (
            {'a': 1, 'b': 2, 'c': 3}, 
            {'a': 1, 'b': 3, 'c': 3, 'd': 4},
            [
                "- b = 3 (changed)",
                "- d = 4 (added)"
            ]
        ),
    ],
)
def test_dict_diff(after, before, expected):
    chain = ProcessValidationChain(
        input_variables=["entities"],
        output_variables=["variables", "result"],
        process=MyProcess,
        memory=ConversationMemory(),
    )
    assert set(chain.variables_diff(before, after)) == set(expected)