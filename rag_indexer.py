import os
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

# CHARGEMENT ET CHUNKING SÉMANTIQUE :

print("1. Chargement et Découpage des Documents ")
try:
    loader = TextLoader("documentation_interne.txt")
    documents = loader.load()
except FileNotFoundError:
    print("ERREUR : Le fichier 'documentation_interne.txt' est introuvable. Assurez-vous qu'il est dans le répertoire courant.")
    exit()

# RecursiveCharacterTextSplitter :

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     
    chunk_overlap=50,   
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Document chargé et découpé en {len(chunks)} fragments (chunks) sémantiques.")

# EMBEDDING ET STOCKAGE PERSISTANT (ChromaDB):

print("\n 2. Création de l'Index Vectoriel (ChromaDB)")

# Initialisation de la fonction d'Embedding ( verifier la necissité d'API Key ):

embedding_function = HuggingFaceEmbeddings()

# Création/Chargement de l'Index:

vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_function, 
    persist_directory="./chroma_db"
)

# Forcer l'écriture sur le disque:
vector_store.persist()
print("✅ Indexation RAG terminée et base de données vectorielle sauvegardée dans './chroma_db'.")

# TEST DE RECHERCHE (Validation):
print("\n 3. TEST DE VALIDATION DE LA RECHERCHE")
query = "Combien de temps faut-il pour finaliser le traitement d'un remboursement ?"
retriever = vector_store.as_retriever(search_kwargs={"k": 1}) 
retrieved_docs = retriever.invoke(query)

if retrieved_docs:
    print(f"Recherche réussie pour la requête : '{query}'")
    print("\nFragments Rétrieves (Chunk) :")
    print(retrieved_docs[0].page_content)
    print("--- FIN DU TEST ---")
else:
    print("⚠️ Aucun document pertinent trouvé lors du test.")