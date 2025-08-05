from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Create two LLM agents using your preferred local model
llm = OllamaLLM(model="llama3")

# Agent A: Researcher Agent
prompt_a = PromptTemplate.from_template(
    "You are a research assistan. Answer this question in detail: {question}"
)
research_agent = prompt_a | llm

# Agent B: Summarizer Agent
prompt_b = PromptTemplate.from_template(
    "You are a communication expert. Briefly summarize the following answer in plain English: {answer}"
)
summarizer_agent = prompt_b | llm

# 1. Ask a complex question
complex_question = "How do AI agents coordinate tasks in multi-agent systems, and what are key challenges?"

# 2. Agent A answers the question
research_answer = research_agent.invoke({"question": complex_question})

# 3. Agent B summarizes Agent A's answer
summary = summarizer_agent.invoke({"answer": research_answer})

print("Research Agent's answer:\n", research_answer.strip())
print("\nSummarizer Agent's plain-language summary:\n", summary.strip())
