from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Agent 1: Give an initial answer
llm = OllamaLLM(model="llama3") # Or your favourite Open model
answer_prompt = PromptTemplate.from_template(
    "Answer as accurately as you can: {question}"
)
answer_agent = answer_prompt | llm

# Agent 2: Critique the answer and suggest improvements
critique_prompt = PromptTemplate.from_template(
    "Here is an answer: {answer}\n\nCritique this answer. Is there anything missing, wrong, or unclear? Suggest how to improve it."
)
critique_agent = critique_prompt | llm

# Agent 3: Re-answer the original question using the critique
improve_prompt = PromptTemplate.from_template(
    "Question: {question}\nOriginal answer: {answer}\nCritique: {critique}\n\nWrite an improved answer that addresses the critique."
)
improve_agent = improve_prompt |  llm

question = "How does retrieval-augmented generation (RAG) improve AI agent accuracy?"


# 1. Initial answer
initial = answer_agent.invoke({"question": question}).strip()

# 2. Critique the answer
critique = critique_agent.invoke({"answer": initial}).strip()

# 3. Produce improved answer
improved = improve_agent.invoke({
    "question": question,
    "answer": initial,
    "critique": critique
}).strip()

print("--- Initial answer ---\n", initial)
print("--- Self-critique ---\n", critique)
print("--- Improved answer ---\n", improved)
