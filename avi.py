import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_openai import ChatOpenAI

# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from langchain_openai import ChatOpenAI
import os


os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key







def getLLamaresponse(input_text, no_words, blog_style):
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ['GROQ_API_KEY'],
        model_name="llama3-8b-8192",
        temperature=0,
        max_tokens=1000,
    )



    # Prompt Template
    template = """
            Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
            """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)

    # Generate the response from the LLama2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional fields

col1, col2 = st.columns([5, 5])  # provinding width of the columns

with col1:
    no_words = st.text_input('Number of words')  # Choose the number of words to generate the blog
with col2:
    blog_style = st.selectbox('Blog Target Audience', ('Researchers', 'Data Scientists', "General"), index=0)

# Button to start generation process
submit = st.button("Generate")

# Final Response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))