import pytest
from lib.ner.ner_prompt_template import NERPromptTemplate  # replace with the actual module and class name

# Parametrized testing with pytest
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "User: Hey\nAI:  Hi! I'm here to help you book an appointment.\n\nWould you be available Sunday 08 at 08:59, Thursday 06 at 16:59 or Friday 07 at 06:59?", 
            "Hi! I'm here to help you book an appointment.\n\nWould you be available Sunday 08 at 08:59, Thursday 06 at 16:59 or Friday 07 at 06:59?"
        ),

        (
            'User: Hi\nAI: Hello\n\nCan you help?\nUser: Of course\nAI: Great! Can you provide details?', 
            'Great! Can you provide details?'
        ),
    ],
)
def test_get_entity_extraction_context(text: str, expected: str):
    print("--------------------")
    print(NERPromptTemplate.get_entity_extraction_context(text))
    print("--------------------")
    assert NERPromptTemplate.get_entity_extraction_context(text) == expected