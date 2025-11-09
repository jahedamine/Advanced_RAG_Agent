import os
import torch
import time
from deep_translator import GoogleTranslator
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uvicorn

# --- Authentification Ngrok ---
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# --- Chargement du corpus ---
FILE_PATH = "documentation_interne.txt"
try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        document_content = f.read()
except FileNotFoundError:
    raise Exception(f"Fichier {FILE_PATH} introuvable.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([document_content])

# --- Embeddings + Vector Store ---
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)
vector_store = FAISS.from_documents(docs, embedding_function)
rag_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- LLM Mistral ---
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512},
    device=0,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# --- Prompt + Chaîne RAG ---
RAG_PROMPT = ChatPromptTemplate.from_template("""
Tu es un assistant IA professionnel francophone. Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question. 
Si la réponse n'est pas dans le contexte, réponds poliment : 'Je suis désolé, cette information spécifique n'est pas disponible dans ma documentation interne.'

--- CONTEXTE ---
{context}
--- FIN CONTEXTE ---

Question: {question}
""")

rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: x['question']) | rag_retriever)
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# --- API FastAPI ---
app = FastAPI(
    title="Advanced RAG Agent API",
    description="API pour poser des questions sur une documentation interne via RAG + Mistral",
    version="1.0.0"
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        raw_response = rag_chain.invoke({"question": query.question})
        if "Assistant:" in raw_response:
            cleaned = raw_response.split("Assistant:")[-1].strip()
        else:
            cleaned = raw_response.strip()
        # Traduction si la réponse est en anglais
        translated = GoogleTranslator(source='auto', target='fr').translate(cleaned)
        return {"answer": translated}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def welcome():
    return {"message": "Bienvenue sur l'API RAG. Utilisez /docs pour tester."}

# --- Lancement Uvicorn + Ngrok ---
def launch_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def launch_ngrok():
    time.sleep(3)
    public_url = ngrok.connect(8000)
    print(f"\n🔗 URL publique de l'API : {public_url}\n")

if __name__ == "__main__":
    print("🚀 Lancement de l'agent RAG + API FastAPI + tunnel Ngrok...")
    api_thread = threading.Thread(target=launch_api)
    ngrok_thread = threading.Thread(target=launch_ngrok)
    api_thread.start()
    ngrok_thread.start()
    api_thread.join()
    ngrok_thread.join()