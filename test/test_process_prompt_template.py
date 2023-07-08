from typing import Optional
import pytest
from lib.process.process_prompt_template import ProcessPromptTemplate
from lib.process.schemas import Process


def test_process_prompt_template_format_with_errors():
    class MyProcess(Process):
        first_name: Optional[str] = None

    template = ProcessPromptTemplate(process=MyProcess, validate_template=False)

    variables = {
        "errors": {"first_name": "error1"},
    }

    kwargs = {
        "variables": variables,
        "history": "User: Hi\nAI: Hi\nUser: Yo\nAI: Yo",
        "diff": []
    }

    output = template.format(**kwargs)
    assert 'Provide the following feedback to the User: "error1"' in output


def test_process_prompt_explain_goals_at_first_turn():
    class MyProcess(Process):
        first_name: Optional[str] = None

    template = ProcessPromptTemplate(process=MyProcess, validate_template=False)

    kwargs = {
        "variables": {},
        "history": "User: Hi",
        "diff": []
    }

    output = template.format(**kwargs)
    assert "Explain your goal to the User" in output


testdata = [
    (
        [{"name": "a", "operation": "updated", "value": 10}],
        'User updated `a`. Confirm the values for `a`.',
    ),
    (
        [{"name": "a", "operation": "added", "value": 10}],
        'User provided `a`. Confirm the values for `a`.',
    ),
    ([], ""),
    (
        [
            {"name": "a", "operation": "updated", "value": 10},
            {"name": "b", "operation": "added", "value": 20},
        ],
        'User provided `b` and updated `a`.',
    ),
]


@pytest.mark.parametrize("data,expected", testdata)
def test_create_output(data, expected):
    class MyProcess(Process):
        first_name: Optional[str] = None

    template = ProcessPromptTemplate(process=MyProcess, validate_template=False)
    result = template.get_updates(data)
    print(result)
    assert result == expected
