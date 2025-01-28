# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 010 | MLOps with AI: H2O + MLflow Agent ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Introduce an AI ML Agent for automating 32+ Machine Learning Models in 30 seconds
# 3. Combine the H2O ML Agent with MLflow for MLOps
# 4. Use MLflow to make predictions in production and manage ML projects (experiments, runs, and artifacts)

# * Project Github: https://github.com/business-science/ai-data-science-team


# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import h2o 
import mlflow
import os
import yaml

from ai_data_science_team.ml_agents import H2OMLAgent, MLflowToolsAgent

# DATA
df = pd.read_csv("data/churn_data.csv")
df

# LLM SETUP
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# Define constants for model
MODEL = "gpt-4o-mini"

# Initialize the language model
llm = ChatOpenAI(model=MODEL)
llm

# 1.0 CREATE THE MACHINE LEARNING AGENT
ml_agent = H2OMLAgent(
    model=llm, 
    enable_mlflow=True, # Use this if you wish to log to MLflow 
)
ml_agent

# RUN THE AGENT
ml_agent.invoke_agent(
    data_raw=df.drop(columns=["customerID"]),
    user_instructions="Please do classification on 'Churn'. Use a max runtime of 30 seconds.",
    target_variable="Churn"
)

# 2.0 CREATE MLflow AGENT
mlflow_agent = MLflowToolsAgent(llm)
mlflow_agent

# what tools do you have access to?
mlflow_agent.invoke_agent(
    user_instructions="What tools do you have access to?",
)
mlflow_agent.get_ai_message(markdown=True)

# launch the mflow UI
mlflow_agent.invoke_agent(user_instructions="launch the mflow UI")
mlflow_agent.get_ai_message(markdown=True)

# what runs are available?
mlflow_agent.invoke_agent(user_instructions="What runs are available in the H2O AutoML experiment?")
mlflow_agent.get_ai_message(markdown=True)

# Make predictions using a specific run ID
mlflow_agent.invoke_agent(
    user_instructions="Make churn predictions on the data set provided using Run ID e070b829ebde401aa7704972cd87ae8a.",
    data_raw=df, # Provide the raw data to the agent for predictions
)
mlflow_agent.get_mlflow_artifacts(as_dataframe=True)

# shut down the mflow UI
mlflow_agent.invoke_agent("shut down the mflow UI")
mlflow_agent.get_ai_message(markdown=True)

