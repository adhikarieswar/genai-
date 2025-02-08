from langchain.agents import Agent
import requests
from bs4 import BeautifulSoup

from searchtool import SearchTool
from webragtool import WebRagTool


def agents(llm):
    # Create agents which uses these tools

    # Has two agents
    # 1. requirements_researcher - search_tool, web_rag_tool
    # 2. swot_analyser

    job_requirements_researcher = Agent(
        role='Market Research Analyst',
        goal='Provide up-to-date market analysis of industry job requirements of the domain specified',
        backstory='An expert analyst with a keen eye for market trends.',
        tools=[SearchTool, WebRagTool],
        verbose=True,
        llm=llm,
        max_iters=1
    )

    resume_swot_analyser = Agent(
        role='Resume SWOT Analyser',
        goal='Perform a SWOT Analysis on the Resume based on (variable) backstory and provide a json report.',
        backstory='An expert in hiring so has a great idea on resumes',
        verbose=True,
        llm=llm,
        max_iters=-1,  # Note: -1 might have a special meaning in your Agent class
        allow_delegation=True
    )
    return job_requirements_researcher, resume_swot_analyser