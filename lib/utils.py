from pydantic import BaseModel, ValidationError


def convert_validation_error_to_dict(error: ValidationError, error_type: str) -> dict:
    error_dict = {}

    for error_item in error.errors():
        field_name = error_item["loc"][0]
        message = error_item["msg"]
        error_type_from_error = error_item["type"]

        if error_type == "missing" and error_type_from_error == "value_error.missing":
            error_dict[field_name] = message
        elif (
            error_type == "assertion" and error_type_from_error != "value_error.missing"
        ):
            error_dict[field_name] = message

    return error_dict


def convert_list_to_string(lst):
    if len(lst) == 0:
        return ""
    elif len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return " and ".join(lst)
    else:
        return ", ".join(lst[:-1]) + " and " + lst[-1]


def dict_diff( after: dict, before: dict,) -> list[dict]:
    changes = []
    filtered_after = {k: v for k, v in after.items() if v is not None}
    filtered_before = {k: v for k, v in before.items() if v is not None}
    for key in filtered_after.keys() - filtered_before.keys():
        if filtered_after[key] is not None:
            changes.append({"name": key, "operation": "added", "value": filtered_after[key]})
    for key in filtered_before.keys() - filtered_after.keys():
        changes.append({"name": key, "operation": "deleted", "value": filtered_before[key]})
    for key in filtered_before.keys() & filtered_after.keys():
        if filtered_before[key] != filtered_after[key]:
            changes.append({"name": key, "operation": "changed", "value": filtered_after[key]})
    return changes

from pydantic.main import BaseModel
from typing import List, Type

def get_fields_with_question(pydantic_model: Type[BaseModel]) -> List[str]:
    schema = pydantic_model.schema()
    fields_with_title = []
    for field_name, field_value in schema['properties'].items():
        if 'question' in field_value:
            fields_with_title.append(field_name)
    return fields_with_title