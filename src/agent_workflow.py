import agents
from crewai import Agent, Task, Crew
from agent_flow import SalesPipeline
import asyncio



if __name__=="__main__":
    (lead_agents_config, lead_tasks_config,
     email_agents_config, email_tasks_config) = agents.read_agents_tasks()

    lead_agents, lead_tasks = agents.define_lead_agents_tasks(
        lead_agents_config, lead_tasks_config)

    # Creating Lead Qualification Crew
    lead_scoring_crew = Crew(
        agents=lead_agents,
        tasks=lead_tasks,
        verbose=True
    )

    email_agents, email_tasks = agents.define_email_agents_tasks(
        email_agents_config, email_tasks_config)

    # Creating Email Crew
    email_writing_crew = Crew(
        agents=email_agents,
        tasks=email_tasks,
        verbose=True
    )

    flow = SalesPipeline(lead_scoring_crew,email_writing_crew)

    emails = asyncio.run(flow.kickoff())