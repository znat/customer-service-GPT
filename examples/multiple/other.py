from langchain import LLMChain, OpenAI, LLMMathChain
from langchain.chat_models import ChatOpenAI

from lib.conversation_memory import ConversationMemory
llm = ChatOpenAI(temperature=0, client=None, max_tokens=200, model="gpt-3.5-turbo")
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.run("What is 13 raised to the .3432 power?")

memory = ConversationMemory()
from langchain.chains.base import Chain
class MasterChain(LLMChain):
    memory: ConversationMemory
    chains: list[tuple[Chain,str]]

prompt = """
I don't know yet"""
PromptTemplate  

mc = MasterChain(
    memory=memory,
    llm=llm_math,
    chains=[(SimpleForm, "simple_form")],

)