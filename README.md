# 🤖 RAG-Based Chatbot using Gemini (Google Generative AI)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Built%20with-Streamlit-brightgreen)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%20API-blueviolet)

---

## 🧠 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot powered by **Google Gemini (Generative AI)**. It enables users to upload PDF documents and interact with a chatbot that provides highly relevant, context-aware answers based on those documents. It combines document parsing, vector search (via ChromaDB), and generative capabilities into one seamless user experience using Streamlit.

---

## 🚀 Key Features

- 📄 **PDF Upload Support**: Upload one or more PDF files to build a knowledge base.
- 🧠 **Vector-Based Document Retrieval**: Uses ChromaDB to perform semantic similarity search using embeddings.
- 🤖 **Gemini-Powered LLM**: Uses **Google Gemini Flash** for fast and accurate natural language responses.
- 🗂️ **Memory Handling**: Maintains session-level chat history.
- 🖥️ **Streamlit UI**: Clean and responsive chat interface with file uploader.

---

## 🧰 Technologies Used

| Tool                         | Purpose                                                  |
|------------------------------|----------------------------------------------------------|
| **Python 3.10+**             | Programming language                                     |
| **Streamlit**                | Web framework for UI                                    |
| **LangChain**                | RAG chain, document loading, and prompt templating      |
| **Chroma**                   | Local vector database to store and query embeddings     |
| **Google Generative AI**     | Gemini API for embeddings and LLM responses             |
| **PyPDFLoader**              | For parsing and loading PDF files                       |

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-chatbot-gemini.git
cd rag-chatbot-gemini
