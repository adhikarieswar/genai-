from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
os.environ['GROQ_API_KEY']='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'
# Load environment variables
load_dotenv()

# Configure Groq API
def get_groq_client():
    return ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",  # Groq API endpoint
        openai_api_key=os.environ['GROQ_API_KEY'],        # Groq API key from environment variable
        model_name="gemma2-9b-it",                        # Llama 3 model
        temperature=0,                                    # Deterministic output
        max_tokens=1000,                                  # Maximum response length
    )

# Function to generate responses using Groq API
def generate_response(prompt):
    llm = get_groq_client()
    response = llm.invoke(prompt)
    return response.content
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidate_data" not in st.session_state:
    st.session_state.candidate_data = {
        "full_name": "",
        "email": "",
        "phone": "",
        "years_of_experience": "",
        "desired_position": "",
        "location": "",
        "tech_stack": [],
    }

# Function to handle the conversation flow
def handle_conversation(user_input):
    # Add user input to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Determine the next step based on missing candidate data
    if not st.session_state.candidate_data["full_name"]:
        st.session_state.messages.append({"role": "assistant", "content": "Great! What is your full name?"})
    elif not st.session_state.candidate_data["email"]:
        st.session_state.messages.append({"role": "assistant", "content": "Could you provide your email address?"})
    elif not st.session_state.candidate_data["phone"]:
        st.session_state.messages.append({"role": "assistant", "content": "What is your phone number?"})
    elif not st.session_state.candidate_data["years_of_experience"]:
        st.session_state.messages.append({"role": "assistant", "content": "How many years of experience do you have?"})
    elif not st.session_state.candidate_data["desired_position"]:
        st.session_state.messages.append({"role": "assistant", "content": "What position are you interested in?"})
    elif not st.session_state.candidate_data["location"]:
        st.session_state.messages.append({"role": "assistant", "content": "What is your current location?"})
    elif not st.session_state.candidate_data["tech_stack"]:
        st.session_state.messages.append({"role": "assistant", "content": "Which technologies, frameworks, or tools are you proficient in? (e.g., Python, Django, React)"})
    else:
        # Generate technical questions based on the tech stack
        tech_stack = ", ".join(st.session_state.candidate_data["tech_stack"])
        prompt = f"Generate 3 technical questions to assess a candidate's proficiency in {tech_stack}."
        questions = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": f"Here are some technical questions for you:\n{questions}"})
        st.session_state.messages.append({"role": "assistant", "content": "Thank you for your time! Our team will review your details and get back to you shortly."})

# Streamlit UI
st.title("TalentScout Hiring Assistant 🤖")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    # Update candidate data based on the conversation
    if not st.session_state.candidate_data["full_name"]:
        st.session_state.candidate_data["full_name"] = user_input
    elif not st.session_state.candidate_data["email"]:
        st.session_state.candidate_data["email"] = user_input
    elif not st.session_state.candidate_data["phone"]:
        st.session_state.candidate_data["phone"] = user_input
    elif not st.session_state.candidate_data["years_of_experience"]:
        st.session_state.candidate_data["years_of_experience"] = user_input
    elif not st.session_state.candidate_data["desired_position"]:
        st.session_state.candidate_data["desired_position"] = user_input
    elif not st.session_state.candidate_data["location"]:
        st.session_state.candidate_data["location"] = user_input
    elif not st.session_state.candidate_data["tech_stack"]:
        st.session_state.candidate_data["tech_stack"] = user_input.split(", ")

    # Handle the conversation
    handle_conversation(user_input)

    # Rerun the app to update the chat interface
    st.rerun()