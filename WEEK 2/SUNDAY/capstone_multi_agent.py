from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

llm = OllamaLLM(model="llama3")

# RAG: Context agent Loads a custom document
loader = TextLoader("domain_knowledge.txt", encoding="utf-8")
docs = loader.load()
embedding = OllamaEmbeddings(model="llama3")
vectordb = Chroma.from_documents(docs, embedding)
rag_qa = RetrievalQA.from_chain_type(
    llm=llm, retriever = vectordb.as_retriever(), chain_type="stuff"
)

# Ethical filter (e.g., block certain keywords)
RESTRICTED = ["violence", "hack", "illegal", "self-harm"]
def ethical_guardrail(question):
    for word in RESTRICTED:
        if word in question.lower():
            return "This topic is blocked by ethical guardrails."
    return None

# Prompts for answering, critiquing, and improving
answer_prompt = PromptTemplate.from_template("Factually answer based on the knowledge base: {q}")
critique_prompt = PromptTemplate.from_template("Critique this answer for bias or errors: {answer}")
improve_prompt = PromptTemplate.from_template(
    "Original Q: {q}\nOriginal answer: {answer}\nCritique: {critique}\nProvide a revised answer."
)
answer_agent = answer_prompt | llm
critique_agent = critique_prompt | llm
improve_agent = improve_prompt | llm

question = "What are best practices for ethical AI development according to this knowledge base?"

if ethical_guardrail(question):
    print(ethical_guardrail(question))
else:
    # RAG step first: gather factual grounding
    rag_fact = rag_qa.invoke({"query": question})["result"]
    # Step 1: Agent answers
    base_answer = answer_agent.invoke({"q": rag_fact}).strip()
    # Step 2: Critique
    critique = critique_agent.invoke({"answer": base_answer}).strip()
    # Step 3: Improved, audited answer
    improved = improve_agent.invoke({
        "q": question,
        "answer": base_answer,
        "critique": critique
    }).strip()
    print("---Initial Answer ---\n", base_answer)
    print("\n--- Critique ---\n", critique)
    print("\n--- Improved & Audited Answer ---\n", improved)
