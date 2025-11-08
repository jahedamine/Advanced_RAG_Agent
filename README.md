# Advanced RAG Agent (Retrieval-Augmented Generation)

Ce dépôt contient le code du **Projet** ADVANCED_RAG_AGENT : Agent RAG avancé capable de répondre à des questions complexes à partir de documents internes (PDF, DOCX, TXT…). Contrairement aux chatbots classiques, il utilise une architecture Retrieval-Augmented Generation (RAG) avec LangChain, recherche vectorielle et LLM pour générer des réponses précises, contextuelles et modulaires. Exposé en API et Dockerisé, il incarne une intelligence documentaire prête à l’emploi.
Le projet a été validé en utilisant l'architecture moderne LCEL (LangChain Expression Language) pour construire une chaîne RAG performante et prouver la capacité du système à filtrer les connaissances générales.

---

## Architecture du Système

Le pipeline RAG est structuré autour de **trois composants principaux**, visant à fournir des réponses précises et contextualisées :

- **Récupération (Retrieval)**  
  Les données sont encodées à l'aide des embeddings `all-MiniLM-L6-v2` (Hugging Face) et stockées dans une base de données vectorielle **FAISS** (utilisée dans Colab).

- **LLM (Large Language Model)**  
  Le modèle `Mistral-7B-Instruct` (via Hugging Face Pipeline sur GPU Colab) est utilisé pour le raisonnement.

- **Chaîne LCEL**  
  Le **LangChain Expression Language** assemble le Retriever et le LLM pour forcer le modèle à répondre uniquement avec le contexte récupéré, validant ainsi la compétence RAG.

---

## Validation du Projet (Google Colab)

En raison des contraintes de mémoire (RAM/VRAM insuffisante pour les gros modèles) sur l'environnement local, le projet a été validé avec succès sur **Google Colab (GPU T4)**.

Le notebook `agent_pipeline_colab.ipynb` prouve la bonne exécution du pipeline à travers **deux tests critiques** :

- **Question Interne (Succès RAG)**  
  Le LLM répond correctement aux questions basées sur le contenu de `documentation_interne.txt`.

- **Question Générale (Échec contrôlé)**  
  Le LLM refuse de répondre à une question hors-sujet, prouvant l'efficacité du mécanisme de filtrage du RAG.

---

## Fonctionnalités
- Chunking + vectorisation locale via FAISS
- Récupération contextuelle avec LangChain
- Génération de réponse via Mistral-7B-Instruct
- Exposition API via FastAPI
- Dockerisation pour déploiement

## Comment Exécuter le Projet

### Fichiers Clés

- `agent_pipeline_colab.ipynb` : Notebook de validation fonctionnel (méthode recommandée)  
- `documentation_interne.txt` : Fichier source de la documentation  
- `requirements.txt` : Liste des dépendances Python

### Instructions Colab

1. Ouvrir le fichier `agent_pipeline_colab.ipynb` dans [Google Colab](https://colab.research.google.com).
2. Activer l'accélérateur matériel **T4 GPU**.
3. Téléverser le fichier `documentation_interne.txt` dans la racine du notebook.
4. Exécuter toutes les cellules séquentiellement.

---

## 🐍 Déploiement local
```bash
pip install -r requirements.txt
uvicorn app:app --reload