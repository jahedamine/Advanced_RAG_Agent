# IMPORTATIONS DES BIBLIOTHÈQUES NÉCESSAIRES :
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool 

# CONFIGURATION DES COMPOSANTS :

# Modèle le plus léger 
LLM_MODEL = "phi3:mini" 

try:
    # On utilise ChatOllama:
    llm = ChatOllama(model=LLM_MODEL, temperature=0) 
    print(f"✅ LLM Agent initialisé avec ChatOllama ({LLM_MODEL}).")
except Exception as e:
    print(f"ERREUR : Impossible d'initialiser ChatOllama. Assurez-vous qu'Ollama est démarré et que le modèle '{LLM_MODEL}' est installé. Erreur: {e}")

# EMBEDDING (RAG - Hugging Face Local et Gratuit):
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} 
)

# CHARGEMENT DU VECTOR STORE (Mémoire RAG):
try:
    vector_store = Chroma(
        persist_directory="./chroma_db_hf", 
        embedding_function=embedding_function
    )
    print("✅ Base de données ChromaDB chargée.")
except Exception as e:
    print(f"ERREUR : Impossible de charger ChromaDB. Avez-vous exécuté rag_indexer.py ? Erreur: {e}")
    exit()

# Création du Retriever:
rag_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# CONSTRUCTION DE LA CHAÎNE RAG DIRECTE (LCEL) :

# Définition du Prompt de RAG (Il prend le contexte ET la question):
RAG_PROMPT = ChatPromptTemplate.from_template("""
Tu es un assistant IA professionnel. Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question. 
Si la réponse n'est pas dans le contexte, tu dois répondre de manière polie : 'Je suis désolé, cette information spécifique n'est pas disponible dans ma documentation interne.'

--- CONTEXTE ---
{context}
--- FIN CONTEXTE ---

Question: {question}
""")

# Assemblage de la Chaîne RAG:
rag_chain = (
    RunnablePassthrough.assign(
        context= (lambda x: x['question']) | rag_retriever 
    )
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# TESTS DE VALIDATION :

print("\n" + "="*50)
print("TEST DE LA CHAÎNE RAG DIRECTE (Compétence RAG validée)")
print("="*50)

# Test 1: Question Interne (DOIT utiliser le RAG pour répondre)
print("\n--- TEST 1: Question Interne ---")
question_interne = "Combien de temps faut-il pour finaliser le traitement d'un remboursement ?"
print(f"Question : {question_interne}")
try:
    response_interne = rag_chain.invoke({"question": question_interne})
    print(f"RÉPONSE RAG : {response_interne}")
except Exception as e:
    print(f"ERREUR D'EXÉCUTION DE LA CHAÎNE : {e}")


# Test 2: Question Généraliste (DOIT répondre 'Information non disponible', car le RAG est forcé)
print("\n--- TEST 2: Question Générale ---")
question_generale = "Quel est le plus grand désert du monde ?"
print(f"Question : {question_generale}")
try:
    response_generale = rag_chain.invoke({"question": question_generale})
    print(f"RÉPONSE RAG : {response_generale}")
except Exception as e:
    print(f"ERREUR D'EXÉCUTION DE LA CHAÎNE : {e}")