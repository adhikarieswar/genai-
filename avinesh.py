


from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama2")
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the user's questions."),
        ("user", "{ques}"),  # Replace with the input variable
        ("assistant", ""),
    ]
)
st.title('Langchain Demo With OΡΕΝΑΙ ΑΡΙ')
input_text=st.text_input("Search the topic u want")


llm=OllamaLLM(model="llama2")

output_parser=StrOutputParser()

chain=prompt|llm|output_parser
if input_text:
    st.write(chain.invoke({'ques':input_text}))









