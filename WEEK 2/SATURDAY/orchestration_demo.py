from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

llm = OllamaLLM(model="llama3");

# Agent 1: Intake - Gathers requirements
intake_prompt = PromptTemplate.from_template(
    "You are an intake agent. Summarize the user's main request in one sentence: {user_input}"
)
intake_agent = intake_prompt | llm

# Agent 2: Planner - Breaks request into steps
plan_prompt = PromptTemplate.from_template(
    "Main request: {summary}\nYou are a planning agent. List up to 3 high-level steps needed to fulfill this request."
)
plan_agent = plan_prompt | llm

# Agent 3: Specialist - Addresses a specific step
specialist_prompt = PromptTemplate.from_template(
    "Step: {step}\nYou are a specialist agent. Briefly explain how to execute this step."
)
specialist_agent = specialist_prompt | llm

# Orchestrator workflow (Centralized pattern)
user_input = "I want to automate weekly financial report generation in my company."


# 1. Intake agent summarizes the request
summary = intake_agent.invoke({"user_input": user_input}).strip()

# 2. Planner generates high-level steps
steps = plan_agent.invoke({"summary": summary}).strip().split("\n")

# 3. Each specialist tackes a step
for step in steps:
    if step.strip():
        detail = specialist_agent.invoke({"step": step.strip()}).strip()
        print(f"Step: {step.strip()}\nHow-to: {detail}\n")
