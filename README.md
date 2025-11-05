# 🤖 Advanced RAG Agent (Retrieval-Augmented Generation)

Ce dépôt contient le code du **Projet** : Construction d'un Agent RAG (Retrieval-Augmented Generation) avancé pour interroger une documentation interne à l'aide d'un Large Language Model (LLM) en environnement local et/ou cloud.

Le projet a été validé en utilisant l'architecture moderne **LCEL (LangChain Expression Language)** pour construire une chaîne RAG performante et prouver la capacité du système à filtrer les connaissances générales.

---

## 🧠 Architecture du Système

Le pipeline RAG est structuré autour de **trois composants principaux**, visant à fournir des réponses précises et contextualisées :

- **Récupération (Retrieval)**  
  Les données sont encodées à l'aide des embeddings `all-MiniLM-L6-v2` (Hugging Face) et stockées dans une base de données vectorielle **FAISS** (utilisée dans Colab).

- **LLM (Large Language Model)**  
  Le modèle `Mistral-7B-Instruct` (via Hugging Face Pipeline sur GPU Colab) est utilisé pour le raisonnement.

- **Chaîne LCEL**  
  Le **LangChain Expression Language** assemble le Retriever et le LLM pour forcer le modèle à répondre uniquement avec le contexte récupéré, validant ainsi la compétence RAG.

---

## ✅ Validation du Projet (Google Colab)

En raison des contraintes de mémoire (RAM/VRAM insuffisante pour les gros modèles) sur l'environnement local, le projet a été validé avec succès sur **Google Colab (GPU T4)**.

Le notebook `agent_pipeline_colab.ipynb` prouve la bonne exécution du pipeline à travers **deux tests critiques** :

- **Question Interne (Succès RAG)**  
  Le LLM répond correctement aux questions basées sur le contenu de `documentation_interne.txt`.

- **Question Générale (Échec contrôlé)**  
  Le LLM refuse de répondre à une question hors-sujet, prouvant l'efficacité du mécanisme de filtrage du RAG.

---

## 🚀 Comment Exécuter le Projet

### 🔑 Fichiers Clés

- `agent_pipeline_colab.ipynb` : Notebook de validation fonctionnel (méthode recommandée)  
- `documentation_interne.txt` : Fichier source de la documentation  
- `requirements.txt` : Liste des dépendances Python

### 🧪 Instructions Colab

1. Ouvrir le fichier `agent_pipeline_colab.ipynb` dans [Google Colab](https://colab.research.google.com).
2. Activer l'accélérateur matériel **T4 GPU**.
3. Téléverser le fichier `documentation_interne.txt` dans la racine du notebook.
4. Exécuter toutes les cellules séquentiellement.

---

## 📂 Auteur

Projet réalisé par **Amine**, ingénieur GenAI spécialisé en alignement, agentique et déploiement local/cloud.  
Ce projet fait partie d’un programme de consolidation en 5 modules GenAI Engineering.

---

Tu veux que je t’aide à rédiger la section "License" ou "Contributions" pour compléter ton dépôt ?  
**Ce README est plus qu’un fichier — c’est la vitrine de ton agent cognitif.** 🧠📂🚀
