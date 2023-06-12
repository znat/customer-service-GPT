from pydantic import ValidationError

def convert_validation_error_to_dict(error: ValidationError, error_type: str) -> dict:
    error_dict = {}

    for error_item in error.errors():
        field_name = error_item['loc'][0]
        message = error_item['msg']
        error_type_from_error = error_item['type']

        if error_type == 'missing' and error_type_from_error == 'value_error.missing':
            error_dict[field_name] = message
        elif error_type == 'assertion' and error_type_from_error != 'value_error.missing':
            error_dict[field_name] = message

    return error_dict

def convert_list_to_string(lst):
    if len(lst) == 0:
        return ''
    elif len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return ' and '.join(lst)
    else:
        return ', '.join(lst[:-1]) + ' and ' + lst[-1]