from agents import agents
from task import Task

def tasks(llm, job_desire, resume_content):
    job_requirements_research = "Find the relevant skills, projects and experience"  # Description (not executed code)
    resume_swot_analysis_description = "Understand the report and the resume based on this make a swot analysis" # Description

    job_requirements_researcher, resume_swot_analyser = agents(llm)  # Get the agents

    research = Task(  # Research Task
        description=f'For Job Position of Desire: {job_desire} research to identify the current market requirements for a person at the job. For searching query use ACTION INPUT KEY as "search_query"',
        expected_output='A report on what are the skills required and some unique real time projects that can be there which enhances the chance of a person to get a job  ',
        agent=job_requirements_researcher  # Use the research agent
    )

    swot_analysis = Task(  # SWOT Analysis Task  <--- NEW TASK DEFINITION
        description=resume_swot_analysis_description,  # Use the description
        expected_output='A SWOT analysis report of the resume in the context of the job requirements ',
        agent=resume_swot_analyser,  # Use the SWOT analysis agent
        input=resume_content  # Pass the resume content as input  <--- IMPORTANT
    )

    # Now you have both tasks defined. You'll likely want to return them
    # or use them in a Crew/Orchestrator to execute them.
    return research, swot_analysis # return both tasks

    # ... (Rest of the tasks.py code to execute or manage the tasks)