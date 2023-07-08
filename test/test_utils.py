import pytest
from lib.utils import dict_diff, get_fields_with_question

@pytest.mark.parametrize(
    "d1, d2, expected",
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
                {'name': 'b', 'operation': 'changed', 'value': 20},
                {'name': 'c', 'operation': 'added', 'value': 30}
            ]
        ),
        (
            {'a': 1, 'b': 2, 'c': 3}, 
            {},
            [
                {'name': 'a', 'operation': 'deleted', 'value': 1},
                {'name': 'b', 'operation': 'deleted', 'value': 2},
                {'name': 'c', 'operation': 'deleted', 'value': 3}, 
            ]
        ),
        (
            {'a': 1, 'b': 2, 'c': 3}, 
            {'a': 1, 'b': 3, 'c': 3, 'd': 4},
            [
                {'name': 'b', 'operation': 'changed', 'value': 3},
                {'name': 'd', 'operation': 'added', 'value': 4}
            ]
        ),
    ],
)
def test_dict_diff(d1, d2, expected):
    assert set(frozenset(d.items()) for d in dict_diff(d1, d2)) == set(frozenset(d.items()) for d in expected)

import pytest
from pydantic import BaseModel, Field
from typing import List

# Your function under test
def get_fields_with_title(pydantic_model: BaseModel) -> List[str]:
    schema = pydantic_model.schema()
    fields_with_title = []
    for field_name, field_value in schema['properties'].items():
        if 'title' in field_value:
            fields_with_title.append(field_name)
    return fields_with_title

# Define Pydantic models for testing
class ModelOne(BaseModel):
    field_one: int = Field(question='Field One')
    field_two: str

class ModelTwo(BaseModel):
    field_one: int
    field_two: str = Field(question='Field Two')

class ModelThree(BaseModel):
    field_one: int = Field(question='Field One')
    field_two: str = Field(question='Field Two')

testdata = [
    (ModelOne, ["field_one"]),
    (ModelTwo, ["field_two"]),
    (ModelThree, ["field_one", "field_two"]),
    (BaseModel, [])
]

@pytest.mark.parametrize("model, expected", testdata)
def test_get_fields_with_title(model, expected):
    assert get_fields_with_question(model) == expected