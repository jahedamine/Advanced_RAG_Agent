# Advanced RAG Agent API (FastAPI + Mistral + LCEL)

Ce dépôt contient le code du **Projet** ADVANCED_RAG_AGENT_API : un agent RAG avancé exposé en API REST, capable de répondre à des questions complexes à partir d’un corpus documentaire interne (`documentation_interne.txt`).  
Contrairement aux assistants classiques, il utilise une architecture **Retrieval-Augmented Generation (RAG)** avec LangChain LCEL, FAISS, embeddings HuggingFace et le modèle Mistral 7B pour générer des réponses précises, contextualisées et filtrées.

---

## Architecture du Système

Le pipeline RAG est structuré autour de **trois composants principaux** :

- **Récupération (Retrieval)**  
  Le corpus est chunké et vectorisé via `all-MiniLM-L6-v2`, puis indexé dans **FAISS**.

- **LLM (Large Language Model)**  
  Le modèle `Mistral-7B-Instruct-v0.2` est chargé localement via HuggingFacePipeline (GPU Colab).

- **Chaîne LCEL**  
  Le **LangChain Expression Language** assemble le Retriever et le LLM pour forcer le modèle à répondre uniquement avec le contexte récupéré, validant ainsi la compétence RAG.

---

## Validation du Projet (Colab + Swagger)

Le projet a été validé dans **Google Colab (GPU T4)** avec exposition via **FastAPI + Ngrok**.  
L’interface Swagger permet de tester l’agent en ligne via `/docs`.

Deux tests critiques ont été réalisés :

- **Question interne (réussite)**  
  → Le modèle répond correctement à une question sur la politique de remboursement.

- **Question hors contexte (filtrage)**  
  → Le modèle refuse poliment de répondre, prouvant l’efficacité du RAG.

---

## Fonctionnalités

- Chunking + vectorisation locale via FAISS
- Récupération contextuelle avec LangChain LCEL
- Génération via Mistral 7B Instruct
- Exposition API via FastAPI
- Tunnel public via Ngrok
- Interface Swagger (`/docs`)
- Nettoyage post-génération (`Assistant:` → réponse propre)
- Dockerisation possible

---

## Fichiers Clés

- `agent_rag_ngrok.py` : Script principal de l’agent RAG exposé en API
- `documentation_interne.txt` : Corpus documentaire interne
- `requirements.txt` : Dépendances Python
- `README.md` : Documentation du projet

---

## Instructions d’Exécution (Colab)

1. Ouvrir le fichier `agent_pipeline_colab.ipynb` dans [Google Colab](https://colab.research.google.com).
2. Activer l'accélérateur matériel **T4 GPU**.
3. Téléverser le fichier `documentation_interne.txt` dans la racine du notebook.
4. Exécuter toutes les cellules séquentiellement.

---

## 🐍 Déploiement local
```bash
pip install -r requirements.txt
uvicorn app:app --reload