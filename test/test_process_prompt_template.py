from typing import Optional
import pytest
from lib.process.process_prompt_template import ProcessPromptTemplate
from lib.process.schemas import Process


def test_process_prompt_template_format_with_errors():
    class MyProcess(Process):
        first_name: Optional[str] = None

    print(ProcessPromptTemplate.__fields__)
    template = ProcessPromptTemplate(process=MyProcess, validate_template=False)

    variables = {
        "errors": {"first_name": "error1"},
    }
    
    kwargs = {
        "variables": variables,
    }

    output = template.format(**kwargs)
    assert "Provide the following feedback to the User: \"error1\"" in output