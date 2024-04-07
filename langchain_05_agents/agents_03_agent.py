from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, load_tools
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from common import constants

llm = OpenAI(openai_api_key=constants.API_KEY, temperature=0.0)

tools = load_tools(["human", "serpapi"], llm=llm, serpapi_api_key=constants.SERP_API_KEY)


prefix = """You are my assistant, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)
print("正在查询西安的天气")
agent_chain.run(input="明天西安雾霾严重吗？然后等待我输入某种天气指标，在根据我的回答给我防护建议，请用中文")
# agent_chain.run(input="根据明天西安的雾霾情况，给我一下你防护建议，请用中文")

