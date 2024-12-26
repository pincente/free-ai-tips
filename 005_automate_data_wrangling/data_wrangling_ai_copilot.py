# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 005 | How To Automate Data Wrangling With AI ----

# PROBLEM: Data wrangling is a time-consuming process that keeps us from analyzing data, making ML models, and getting business insights.

# GOALS: 
# - Expose you to my new AI Data Science Team of Copilots
# - Create an AI Copilot to automate data wrangling
# - Use the AI Copilot to wrangle a folder of excel files

# Project Github: https://github.com/business-science/ai-data-science-team

# Installation: pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

# Libraries

from langchain_openai import ChatOpenAI

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from pprint import pprint

from ai_data_science_team.agents import make_data_wrangling_agent

# 1.0 SETUP 

# PATHS
PATH_ROOT = "005_automate_data_wrangling/"

# LLM
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

MODEL    = "gpt-4o-mini"

# LOGGING
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")

# READ DATA

files = []
for file in os.listdir(os.path.join(os.getcwd(), PATH_ROOT, "data/")):
    print(file)
    files.append(pd.read_csv(os.path.join(os.getcwd(), PATH_ROOT, "data/", file)).to_dict())
    
files

# 2.0 CREATE AI COPILOT

# Create the AI Copilot

llm = ChatOpenAI(model = MODEL)

data_wrangling_agent = make_data_wrangling_agent(model = llm, log=LOG, log_path=LOG_PATH)

data_wrangling_agent

# 3.0 WRANGLE DATA

# Run data wrangling agent on the data

response = data_wrangling_agent.invoke({
    "user_instructions": "This is a folder of vehicle data by manufacturer. Please combine each file into a single data frame. Assess for any additional wrangling that may be needed.",
    "data_raw": files,
    "max_retries":3,
    "retry_count":0
})

response.keys()

# 4.0 RESULTS

response["data_wrangled"]

pprint(response['messages'][0].content)
