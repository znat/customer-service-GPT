# 🧑 Customer-Service-GPT

A basic toolkit to experiment with LLM-powered process-driven chatbots. It is Experimental and not meant to be used in a prod setup and is based on [Pydantic](https://github.com/pydantic/pydantic) and [Langchain](https://github.com/hwchase17/langchain)

Process-driven chatbots assist users in completing tasks by guiding them through a sequence of steps, such as opening a bank account or scheduling an appointment.

This expermiments with:
### Contextual named entity recognition

Example:
```
AI: Hello! I'm here to help you book an appointment at our hair salon. To get started, could you please let me know your availability? We have the following options: Thursday 20 at 23:45, Saturday 15 at 03:45, or Wednesday 19 at 00:45.
USER: last one is great

entities:
- availability: Wednesday 19 at 00:45
```

### Dialogue state management

The `Process` which collects data from the user (an approach borrowed from [Rasa forms](https://rasa.com/docs/rasa/forms/)).
The experiment consists here in using the business logic to format the prompt such that
the model has the essential information to "reason" and formulate a next AI message.

Currently it works with `gpt-3.5-turbo` and shines with `gpt-4`.

- [🧑 Customer-Service-GPT](#-customer-service-gpt)
    - [Contextual named entity recognition](#contextual-named-entity-recognition)
    - [Dialogue state management](#dialogue-state-management)
  - [👷 Install](#-install)
  - [🎬 Demos](#-demos)
    - [📅 Appointment booking](#-appointment-booking)
    - [💇 Duplex clone](#-duplex-clone)
  - [⚙️ Quick start](#️-quick-start)
    - [1. Define a process](#1-define-a-process)
    - [2. Define entities](#2-define-entities)
    - [3. Instantiate a `ProcessChain`](#3-instantiate-a-processchain)
  - [Using the `ProcessChain`](#using-the-processchain)
    - [`process_description`](#process_description)
    - [Process completion and `Result`](#process-completion-and-result)
    - [Validation and working variables](#validation-and-working-variables)
      - [Validating a single field](#validating-a-single-field)
      - [Cross-valudation and working variables](#cross-valudation-and-working-variables)
      - [Interpolation](#interpolation)
  - [Using the `NERChain`](#using-the-nerchain)
    - [Basics](#basics)
      - [Value and context](#value-and-context)
      - [DateTimeEntity (using LLMs to parse entities)](#datetimeentity-using-llms-to-parse-entities)


## 👷 Install

Requirements: python 3.10 and [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
```bash
git clone https://github.com/znat/customer-service-GPT/
poetry install
```
Create an `.env` file at the root containing your OpenAI API key. 
```
OPENAI_API_KEY=<key>
```

## 🎬 Demos

### 📅 Appointment booking

An AI medical assistant helps users book appointments at the clinic by trying to find matching availability, answering questions, and collecting necessary information to confirm the appointment

```bash
poetry run python -m examples.appointment_booking.example_booking_bot
```

<div align="center">
<img align="center" src="./examples/appointment_booking/appointment_booking_demo.gif" alt="demonstration" width=500>
</div>

### 💇 Duplex clone

This bot replicates the Google Duplex demo, in which an AI schedules a hair salon appointment. The AI is the customer assistant and the User is the hair salon attendant.
```bash
poetry run python -m examples.appointment_booking.example_duplex_bot
```


<div align="center">
<img align="center" src="./examples/appointment_booking/google_duplex_demo.gif" alt="demonstration" width=500>
</div>

## ⚙️ Quick start

The `ProcessChain` is a sequential [Langchain](https://github.com/hwchase17/langchain) chain working as follows:

1. `NERChain` extracts entities from the user response based on examples
2. `ValidationChain` stores entity values in variables if they are valid
3. `ConversationChain` evaluates the context and asks the next question or provides feedback.

The `ProcessChain` outputs a `result` object with the information collected when the task is completed.

### 1. Define a process
The process is a pydantic model where fields are described.
The fields will be collected during the process using the questions provided, validated with the validators and the errors messages will be surfaced in the conversation.

```python
class SimpleForm(Process):
    process_description = """
You are a retail bank account executive AI.
Your goal is to collect the required information from the User to open a bank account.
"""
    first_name: Optional[str] = Field(
        name="First name",
        description="First name of the user, required to open an account",
        question="What is your first name?",
    )

    age: Optional[int] = Field(
        name="Age",
        description="Age of the user, required to open an account",
        question="What is your age?",
    )

    @validator("first_name")
    def validate_first_name(cls, v):
        assert v[0].isalpha(), "First name must start with a letter."
        return v.capitalize()

    @validator("age")
    def validate_age(cls, v):
        assert v is None or v >= 18, "Age must be 18 or older"
        return v
```
### 2. Define entities

Although entities generally match form fields, they are not the same. Entities are extracted from the user input, then processed by the `ValidationChain` which may decide to save values to variables.

```python
examples = [
    {"text": "I'm Nathan", "entities": [{"name": "first_name", "value": "Nathan"}]},
    {
        "text": "I'm jenny and I'm 98 yo",
        "entities": [
            {"name": "first_name", "value": "Jenny"},
            {"name": "age", "value": 98},
        ],
    },
    {
        "text": "I'm Jo Neville and I'm a plumber",
        "entities": [
            {"name": "first_name", "value": "Jo"},
            {"name": "last_name", "value": "Neville"},
        ],
    },
    {"context": "How old are you?", "text": "Yes", "entities": []},
]
```
Notice that the `value` might not be related to a particular substring in the `text`. Observe the last two examples which aim to capture a confirmation based on a yes or no. We want the entity to be extracted only in the `context` of confirming data, but not if the `context` of answering a specific question.

### 3. Instantiate a `ProcessChain`
The `ProcessChain` glues everything together.

```python
openai_entity_extractor_llm = ChatOpenAI(temperature=0, client=None, max_tokens=256)
openai_chat_llm = ChatOpenAI(temperature=0, client=None, max_tokens=256, model="gpt-4) # Note: wor

process_chain = ProcessChain(
    ner_llm=openai_entity_extractor_llm,
    chat_llm=openai_chat_llm,
    entities={
        "first_name": Entity,
        "age": IntEntity,
        "last_name": Entity,
        "confirmed": BooleanEntity,
    },
    entity_examples=[EntityExample.parse_obj(e) for e in examples],
    form=MyForm,
    memory=ConversationMemory(),
    verbose=True,
)
```

Then launch the bot and open the url provided:

```python
gradio_bot(chain=process_chain, initial_input="hey", title="FormBot").launch() # Hey is a trick to get the bot to start the conversation as it normally reacts to a user input
```

See code examples for more details and more complex entities such as dates.

## Using the `ProcessChain`

### `process_description`

A process contains a `process_description` that will be injected in the prompt. Here is an example:

```python
    process_description = f"""
You are a hair salon attendant AI. The User wants to book and appointment

You can share the following information with the User if the User asks:
- Address of the salon is 123 Main Street, CoolVille, QC, H3Z 2Y7
- The salon phone number is 514-666-7777

To all other questions reply you don't know. 

"""
```

### Process completion and `Result`

A process yield a completion `Result` so the `ProcessChain` can be invoked by other chain. The `ProcessChain` can execute in a `while` loop until a `Result` object is output.

To ouput a result you can use the two following `Process` methods:

```python
class MyProcess(Process):
    ...
    def is_completed(self) -> bool:
        return ... # A sucess condition, e.g a confirmation set to `True`
    
    def is_failed(self) -> bool:
        return ... # A failure condition, e.g an error counter reaching a certain value
```

### Validation and working variables

A `Process` is a `pydantic` model and thus leverages built-in validation methods.

#### Validating a single field

One way to validate is to make an assertion in a field `@validator` function.
The error message will be surfaced to the user. 

```python
phone_number: Optional[str] = Field(
    name="Phone number",
    description="Phone number of the user, required to contact a patient in case of unexpected events, such as delays or cancellations",
    question="What is your phone number?",
)

@validator("phone_number")
def validate_phone_number(cls, v):
    import re

    pattern = re.compile(r"^\d{3}-\d{3}-\d{4}$")
    assert v is None or pattern.match(
        v
    ), f"Can you please provide a valid phone number using the XXX-XXX-XXXX format?"
    return v
```

Note that you can also interpolate variables in the error message as follows: `{{variables}}`

#### Cross-valudation and working variables

If you need to use another variable for your validation or want to set a variable based on the content of another you can use the pydantic `@root_validator`:


```python
class MyProcess(Process):

    # This field has a question, so the question will be asked to the user
    availability: Optional[dict | str] = Field(
        title="Availability for doctor appointment",
        question="Would you be available {{matching_slots_in_human_friendly_format}}",
        description=f"Providing availability helps finding an available slot in the salon's calendar",
    )

    # This field has NO question, so it will not be surfaced by the user
    # but can used as a working/internal processing variable.
    appointment_time: Optional[str] = Field(
        title="Appointment in human frendly format",
        aknowledgement="Great, we have booked an appointment on {{human_friendly_appointment_time}}",
    )
    @root_validator(pre=True)
    def validate(cls, values: dict):
        ...
        # Let's define a default value in case no availability is provided yet or
        # there is no matching slot between the user availability and the salon availability
        values["appointment_time"] = ... # Add some logic to set appointement time
        ...
        return values
```

If you want to surface validation errors but still need the object to be successfully instantiated so that updated variables are available in the conversation or can be interpolated, you can also add errors as follows in the `@root_validator`

```python
    @root_validator(pre=True)
    def validate(cls, values: dict):
        ...
        values["_errors"] = {
            # the key is the variable for which the feedback is given
            "availability": "Sorry, we're not available at these dates, but we can offer {{other_slots}}
        }
        ...
        return values
```

This should be surfaced to the user in the next AI message.

#### Interpolation

All variables can be interpolated in questions. For example, the `appointment_time` could be used later in the process.

```python
class MyProcess(process):
    ...
    # This var is set by the validator.
    appointment_time: Optional[str] = Field(
        title="Appointment in human frendly format",
    )

    # And can be used in another question
    confirm: Optional[str] = Field(
        title="Do you confirm the appointment on {{appointment_time}}?",
    )
```

## Using the `NERChain`

### Basics

The `NERChain` is a simple few-shots named entity regognition. It is used by the `ProcessChain` but you can also experiment directly with it.

```python
examples =[
    {
        "text": "I'm Jo Neville and I'm a plumber",
        "entities": [
            {"name": "first_name", "value": "Jo"},
            {"name": "last_name", "value": "Neville"},
            {"name": "occupation", "value": "plumber"},
        ],
    },
    ...
]

entities = {
    "first_name": Entity,
    "last_name": Entity,
    "occupation": Entity,
    "email": EmailEntity,
}

ner_chain = NERChain(
    llm=entity_extractor_llm,
    additional_instructions="""
"confirmation" entity should only be `true` if the gives an explicity confirmation that all the collected information is correct
and not if the user just says "yes" or confirms but asks follow-up questions.
    """,
    verbose=True,
    entities=entities,
    examples=[EntityExample.parse_obj(e) for e in examples],
)
```

#### Value and context
The context is the last bot utterance prior the user utterance were entities are extracted,
A benefit of LLMs is they pickup simple transformations quite well. The entity value does not need to be 
a substring of the user utterance (`text`). In the following example, the context helps the model understand what 
*"The second option"* is referring to

```yaml
- context: 'I can propose Wednesday 14 at 09:00, Friday 16 at 11:00 or Sunday 18 at 11:00.'
  text: 'the second option is great'
  entities:
      - name: availability
        value: 'Friday 16 at 11:00'
```

#### DateTimeEntity (using LLMs to parse entities)

The `DateTimeEntity` is an illlustration of how an LLM can be use for more rigourous and structured transformation.
`DateTimeEntity` will output an ISO formatted timespan. Here are a few examples:

```
text: "Any availability next week?"
entities:
[
   {
      "name":"availability",
      "value":{
         "start":"2023-06-26T00:00:00",
         "end":"2023-07-02T23:59:59",
         "grain":604800
      }
   }
]
```

You can try it with:
```
poetry run python example_ner_chain.py
```