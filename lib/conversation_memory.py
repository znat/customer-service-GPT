from typing import Any, Dict, List
from langchain.schema import BaseMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.memory.buffer_window import ConversationBufferWindowMemory


class KeyValueStoreMemory(BaseMemory):
    memory_key: str = "variables"
    memories: dict[str, Any] = dict()

    def get(self, key: str) -> Any:
        return self.memories.get(key)

    def set(self, key: str, value: Any) -> None:
        self.memories[key] = value

    def delete(self, key: str) -> None:
        if key in self.memories.keys():
            del self.memories[key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {self.memory_key: self.memories}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        variables = outputs.get("variables", {})
        for k, v in variables.items():
            self.set(k, v)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def clear(self) -> None:
        return self.memories.clear()

import traceback
class ConversationHistoryMemory(ConversationBufferWindowMemory):
    human_prefix: str = "User"
    memory_key: str = "history"
    history: list[HumanMessage | AIMessage] = []

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def clear(self) -> None:
        return self.history.clear()
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str = inputs.get("input")
        output_str = outputs.get("response")
        if input_str:
            self.chat_memory.add_user_message(input_str)
        if output_str:
            self.chat_memory.add_ai_message(output_str)


class ConversationMemory(BaseMemory):
    kv_store: KeyValueStoreMemory = KeyValueStoreMemory()
    history: ConversationHistoryMemory = ConversationHistoryMemory()

    @property
    def memory_variables(self) -> List[str]:
        return self.kv_store.memory_variables + self.history.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **self.kv_store.load_memory_variables(inputs),
            **self.history.load_memory_variables(inputs),
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if "variables" in outputs.keys():
            self.kv_store.save_context(inputs, outputs)
        else:
            self.history.save_context(inputs, outputs)

    def clear(self) -> None:
        self.kv_store.clear()
        self.history.clear()
