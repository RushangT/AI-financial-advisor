import streamlit as st
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    st.error("Error: Missing API keys. Please check your environment variables.")
    st.stop()

# Initialize the LangGraph workflow
from langgraph.graph import START, END
from typing import TypedDict, Annotated
from typing_extensions import operator

class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    messages: Annotated[list[AnyMessage], operator.add]

# Define the workflow graph
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import ChatOpenAI
from langchain_experimental.utilities import PythonREPL

search = TavilySearchResults(max_results=2)
tools = [YahooFinanceNewsTool(), search, PythonREPL()]
llm = ChatOpenAI(model="gpt-4o")

workflow = StateGraph(GraphState)

def reasoner(state):
    query = state["query"]
    messages = state["messages"]
    sys_msg = f"You are an expert financial advisor. Use tools like news sentiment analysis and stock price lookup to provide actionable financial advice."
    message = {"role": "user", "content": query}
    messages.append(message)
    return {"messages": messages, "result": f"Response for {query}"}  # Placeholder logic

workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "reasoner")
workflow.add_conditional_edges("reasoner", tools_condition)
workflow.add_edge("tools", "reasoner")
react_graph = workflow.compile()

# Streamlit Page
st.set_page_config(page_title="Financial Advisor", page_icon="💰", layout="centered")

# Page Title and Styling
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f9fa;
    }
    .main-title {
        font-size: 36px;
        color: #343a40;
        text-align: center;
        margin-bottom: 20px;
    }
    .container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .result {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Financial Advisor Chatbot</div>', unsafe_allow_html=True)

# Input Container
with st.container():
    query = st.text_input("Enter your financial query:", placeholder="E.g., Should I invest in Google?")
    if st.button("Submit"):
        if query.strip():
            with st.spinner("Analyzing your query..."):
                response = react_graph.invoke({"query": query, "messages": []}, config={"recursion_limit": 100})
                st.session_state["last_query"] = query
                st.session_state["response"] = response
        else:
            st.warning("Please enter a query before submitting.")

# Output Container
if "response" in st.session_state:
    with st.container():
        st.markdown('<div class="container">', unsafe_allow_html=True)
        st.subheader("Query:")
        st.write(st.session_state["last_query"])
        st.subheader("Advice:")
        st.markdown(f"<div class='result'>{st.session_state['response']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <footer style="text-align: center; margin-top: 20px;">
        <small>💡 Powered by LangGraph, OpenAI, and Streamlit</small>
    </footer>
    """,
    unsafe_allow_html=True,
)