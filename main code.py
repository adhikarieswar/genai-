from crewai import Crew, Process
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq  # Mixtral
from utils import read_all_pdf_pages,fitz
from agents import agents
from tasks import tasks
import os
load_dotenv(find_dotenv())

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

resume = read_all_pdf_pages("C:\Users\DELL\Desktop\new_important_resume_avinash.pdf")
job_desire = input("Enter Desiring Job: ")

job_requirements_researcher, resume_swot_analyser = agents(llm)

research, resume_swot_analysis = tasks(llm, job_desire, resume)

crew = Crew(
    agents=[job_requirements_researcher, resume_swot_analyser],
    tasks=[research, resume_swot_analysis],
    verbose=1,
    process=Process.sequential
)
print(crew.kickoff()) 