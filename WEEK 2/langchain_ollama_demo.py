from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# 1. Connect to your Ollama LLM (local model running via ollama serve)
llm = OllamaLLM(model="llama3")  # Or use "mistral" if you prefer

# 2. Define your prompt template
prompt_template = PromptTemplate.from_template(
    "You are an assistant who answers concisely. Question: {q}"
)

# 3. Chain prompt → LLM using the pipe operator
agent = prompt_template | llm

# 4. Prepare a question and invoke the agent
question = "What are some ethical challenges of AI agents in healthcare?"
response = agent.invoke({"q": question})

# 5. Output the answer
print("Assistant's answer:", response.strip())
