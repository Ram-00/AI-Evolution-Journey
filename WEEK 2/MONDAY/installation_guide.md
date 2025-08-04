# 🛠️ LangChain + Ollama + Open-Source LLM: Install & Setup Guide

This guide walks you through setting up a **local, subscription-free AI agent** using [LangChain](https://www.langchain.com/) and an open-source LLM (e.g., Llama 3 or Mistral) via [Ollama](https://ollama.com/). No paid APIs, no vendor lock-in—just pure open-source innovation.

---

## 1. Prerequisites

- **Hardware:**  
  - Windows, Mac, or Linux computer
  - At least 8GB RAM (more if using larger models)
- **Software:**  
  - [Python](https://www.python.org/) 3.9 or above  
  - [pip](https://pip.pypa.io/en/stable/)

---

## 2. Install Python (if not already)

- Download from [python.org](https://www.python.org/downloads/).
- During install, select **"Add Python to PATH"**.
- Verify with:

    ```
    python --version
    pip --version
    ```

---

## 3. Install Ollama (Local LLM Host)

- Go to [Ollama Download](https://ollama.com/download) and select your OS.
- Run the installer (follow on-screen instructions).
- Start the Ollama server (only once, in a terminal/PowerShell):

    ```
    ollama serve
    ```

  > **Tip:** If you see a port error, Ollama is already running—no need to start it again.

---

## 4. Download an Open-Source LLM Model

- Choose one (or several); for most users, Llama 3 or Mistral is ideal:

    ```
    ollama pull llama3
    ollama pull mistral
    ```

- Check available models:

    ```
    ollama list
    ```

---

## 5. Set Up Your Python Project Environment

1. **(Recommended)** Create a new directory for your project.
2. **Optional but good practice:** Set up a virtual environment

    ```
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate
    ```

---

## 6. Install LangChain and Integration Libraries
```
pip install -U langchain langchain-ollama
```

> If you encounter install issues, be sure your pip and setuptools are up to date:  
> `pip install --upgrade pip setuptools`

---

## 7. Save and Run the Example Code

1. Create a file like `langchain_ollama_demo.py` and paste the following code:

    ```
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate

    # 1. Connect to your Ollama LLM
    llm = OllamaLLM(model="llama3")  # Or "mistral" if you prefer

    # 2. Prepare your prompt
    prompt_template = PromptTemplate.from_template(
        "You are an assistant who answers concisely. Question: {q}"
    )

    # 3. Compose prompt → LLM
    agent = prompt_template | llm

    # 4. Ask a question
    question = "What are some ethical challenges of AI agents in healthcare?"
    response = agent.invoke({"q": question})

    # 5. Output the answer
    print("Assistant's answer:", response.strip())
    ```

2. **Run the script:**

    ```
    python langchain_ollama_demo.py
    ```

3. **Expected output:**  
   A concise list of ethical challenges directly from your chosen local LLM.

---

## 8. Troubleshooting

- **Ollama server port error:**  
  Means Ollama is already running. Just proceed.
- **`ModuleNotFoundError`:**  
  Double-check your `pip install` commands and Python environment.
- **`TypeError: string indices must be integers, not 'str'`:**  
  Just print `response.strip()`, not `response["text"]`.

---

## 9. What Next?

- Try other prompts or switch models (change `"llama3"` to `"mistral"`).
- Experiment with adding tools, chains, or datasets.
- Share your setup and code—this guide can be bundled as `installation_guide.md` on GitHub for your learning repo!

---

*Ready to build open, ethical, and powerful AI agents—at zero cloud cost? Let’s keep innovating, the open-source way!*
