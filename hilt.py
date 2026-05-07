from langgraph.graph import StateGraph, START , END 
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage , AIMessage , AnyMessage
from typing import TypedDict , Literal  , Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt , Command
from dotenv import load_dotenv
load_dotenv()

llm = ChatNVIDIA(model="nvidia/nemotron-mini-4b-instruct", temperature=0)

from langgraph.graph.message import add_messages

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question.",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })
    
    if decision["approved"] == 'no':
        return {"messages": [AIMessage(content="Not approved.")]}

    else:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Checkpointer is required for interrupts
checkpointer = MemorySaver()


app = builder.compile(checkpointer=checkpointer)
app


# Create a new thread id for this conversation
config = {"configurable": {"thread_id": '1234'}}

initial_input = {
    "messages": [
        ("user", "Explain what is Bus,")
    ]
}

# Invoke the graph 
result = app.invoke(initial_input, config=config)
result

message = result['__interrupt__'][0].value
message

user_input = input(f"\nBackend message = {message} \n Approve this question (y/n): ")

final_result = app.invoke(
    Command(resume = {"approved" : user_input}),
    config = config
)

print(final_result)