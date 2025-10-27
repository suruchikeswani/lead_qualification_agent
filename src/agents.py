import os
import yaml
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from response_struct import LeadScoringResult

load_dotenv()

os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'

def read_agents_tasks():
    files = {
        'lead_agents': 'config/lead_qualification_agents.yaml',
        'lead_tasks': 'config/lead_qualification_tasks.yaml',
        'email_agents': 'config/email_engagement_agents.yaml',
        'email_tasks': 'config/email_engagement_tasks.yaml'
    }

    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    lead_agents_config = configs['lead_agents']
    lead_tasks_config = configs['lead_tasks']
    email_agents_config = configs['email_agents']
    email_tasks_config = configs['email_tasks']
    return lead_agents_config,lead_tasks_config,email_agents_config,email_tasks_config

def define_lead_agents_tasks(lead_agents_config, lead_tasks_config):
    agents = []
    lead_data_agent = Agent(
        config=lead_agents_config['lead_data_agent'],
        tools=[SerperDevTool(), ScrapeWebsiteTool()]
    )
    agents.append(lead_data_agent)

    cultural_fit_agent = Agent(
        config=lead_agents_config['cultural_fit_agent'],
        tools=[SerperDevTool(), ScrapeWebsiteTool()]
    )
    agents.append(cultural_fit_agent)

    scoring_validation_agent = Agent(
        config=lead_agents_config['scoring_validation_agent'],
        tools=[SerperDevTool(), ScrapeWebsiteTool()]
    )
    agents.append(scoring_validation_agent)

    tasks = []
    lead_data_task = Task(
        config=lead_tasks_config['lead_data_collection'],
        agent=lead_data_agent
    )
    tasks.append(lead_data_task)

    cultural_fit_task = Task(
        config=lead_tasks_config['cultural_fit_analysis'],
        agent=cultural_fit_agent
    )
    tasks.append(cultural_fit_task)

    scoring_validation_task = Task(
        config=lead_tasks_config['lead_scoring_and_validation'],
        agent=scoring_validation_agent,
        context=[lead_data_task, cultural_fit_task],
        output_pydantic=LeadScoringResult
    )
    tasks.append(scoring_validation_task)

    return agents, tasks

def define_email_agents_tasks(email_agents_config, email_tasks_config):
    agents = []
    email_content_specialist = Agent(
        config=email_agents_config['email_content_specialist']
    )
    agents.append(email_content_specialist)

    engagement_strategist = Agent(
        config=email_agents_config['engagement_strategist']
    )
    agents.append(engagement_strategist)

    tasks = []
    email_drafting = Task(
        config=email_tasks_config['email_drafting'],
        agent=email_content_specialist
    )
    tasks.append(email_drafting)
    engagement_optimization = Task(
        config=email_tasks_config['engagement_optimization'],
        agent=engagement_strategist
    )
    tasks.append(engagement_optimization)

    return agents, tasks
