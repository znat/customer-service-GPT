import json
from langchain.chains.base import Chain
from rich.console import Console
from rich.prompt import Prompt

from lib.conversation_memory import ConversationMemory

from .process.process_chain import ProcessChain


def console_bot(chain: ProcessChain, initial_input: str = ""):
    console = Console()
    console.clear()
    console.print(chain(initial_input)["response"])
    while True:
        user_input = Prompt.ask("User")
        if user_input.lower() == "exit":
            console.print("[bold]Goodbye![/bold]")
            break
        else:
            inference = chain(user_input)
            console.print(inference["response"])
            if inference["result"] is not None:
                console.print("[bold]Done![/bold]")
                console.print(inference["result"])
                break

# def console_popcorn(chains: list[ProcessChain], initial_input: str = ""):
#     console = Console()
#     console.clear()
#     console.print(chain(initial_input)["response"])
#     while True:
#         user_input = Prompt.ask("User")
#         if user_input.lower() == "exit":
#             console.print("[bold]Goodbye![/bold]")
#             break
#         else:
#             inference = chain(user_input)
#             console.print(inference["response"])
#             if inference["result"] is not None:
#                 console.print("[bold]Done![/bold]")
#                 console.print(inference["result"])
#                 break


from langchain.callbacks.base import BaseCallbackHandler

"""Callback Handler that prints to std out."""
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.schema import AgentAction, AgentFinish, LLMResult
from rich.console import Console
from rich.prompt import Prompt


class RichCallbackHandler(BaseCallbackHandler):
    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color
        self.console = Console()

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        self.console.log("on_text")
        self.console.print(text)


def gradio_bot(chain: Chain, initial_input: str = "", title: str = "CustomerServiceGPT"):

    
    import gradio as gr
    css = """
#chatbot .user {
    text-align: right
}
"""
    with gr.Blocks(css=css, theme="lightdefault") as demo:
        gr.Markdown(
            f"""
        # {title}
        """
        )
        initial_history = [[None, chain(initial_input)["response"]]] if initial_input else []
        chatbot = gr.Chatbot(
            value=initial_history, elem_id="chatbot"
        ).style(height=500)
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                ).style(container=False)
            with gr.Column(scale=1):
                clear = gr.Button("Reset conversation")

        def respond(message, chat_history):
            inference = chain(message)
            response = inference["response"]
            if inference["result"] is not None:
                response +=  f"""

```json
{json.dumps(inference["result"], indent=2)}
```
"""
            chat_history.append((message, response))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot]) # type: ignore

        def reset():
            chain.reset()
            return [[None, chain(initial_input)["response"]]]

        clear.click(reset, None, chatbot, queue=False)

    return demo
