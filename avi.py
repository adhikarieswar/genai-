import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper








def getLLamaresponse(input_text, no_words, blog_style):

    llm = CTransformers(model="meta-llama/Llama-2-7b-hf",model_path='https://huggingface.co/meta-llama/Llama-2-7b-chat-hf',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})


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