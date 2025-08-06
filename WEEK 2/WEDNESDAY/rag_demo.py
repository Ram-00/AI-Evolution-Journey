from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# Step 1: Prepare your vector DB (Chroma) with sample content
loader = TextLoader("sample_knowledge.txt")
docs = loader.load()

# Step 2: Build the embedding retriever from Ollama (Local LLM)
embedding = OllamaEmbeddings(model="llama3")
vectordb = Chroma.from_documents(docs, embedding)

# Step 3: Set up the Retriever + LLM pipeline (RAG)
llm = OllamaLLM(model="llama3")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

# Step 4: Ask a knowledge-based question!
query = "What were the major milestones in the development of cricket in India from its introduction to the modern era?"
response = qa_chain.invoke({"query": query})

print("RAG-augmented agent answer:\n", response["result"].strip())
