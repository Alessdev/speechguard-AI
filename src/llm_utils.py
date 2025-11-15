from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def get_chat_model(temperature: float = 0.1):
    return Ollama(
        model="phi3:mini",
        temperature=temperature,
    )

def get_embeddings_model():
    # NO se requiere embeddings en este proyecto,
    # puedes dejarlo vacío o simularlo si fuera necesario.
    return None

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_chain(prompt_template_str: str):
    llm = get_chat_model()
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    return prompt | llm

