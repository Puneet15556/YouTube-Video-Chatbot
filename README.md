# 🎥 YouTube Video Chatbot with Contextual Question Answering

An intelligent chatbot that allows users to **ask questions about any YouTube video** and receive **context-aware, accurate answers** based on the video’s transcript. This application leverages **LLMs**, **semantic search**, and **RAG (Retrieval-Augmented Generation)** to enable smart and efficient question answering in real time.

---

## 🚀 Project Overview

This project bridges YouTube content with conversational AI by extracting and processing transcripts, semantically indexing them, and enabling intelligent Q&A interactions. It demonstrates the practical integration of modern **language models**, **embedding-based search**, and **real-time web applications**.

---

## 🧠 Key Features

- 🔎 **Transcript Extraction**: Automatically retrieves video transcripts using the **YouTube Transcript API**.
- 🧩 **Chunking & Preprocessing**: Splits long transcripts into meaningful segments for efficient indexing.
- 🧠 **Embeddings & Vector Store**:
  - Uses **OpenAI/Instructor/HuggingFace embeddings**
  - Creates a **FAISS index** for fast semantic search
- 💬 **LLM-Powered QA**:
  - Retrieves relevant context
  - Generates answers using **LLMs** (like GPT/Langchain-Groq)
  - Ensures responses are grounded in video content
- 🌐 **Streamlit Interface**:
  - Simple UI to input YouTube Video ID and questions
  - Displays retrieved context and answers

---

## 🧰 Tech Stack

| Component | Tool/Library |
|----------|---------------|
| Programming | Python |
| Transcript API | youtube-transcript-api |
| Embeddings | OpenAI / HuggingFace |
| Vector Store | FAISS |
| QA Model | OpenAI GPT / LangChain-Groq |
| Frontend | Streamlit |
| RAG Pipeline | LangChain |

---

## 🛠️ How it Works

1. **User Inputs**: YouTube Video ID and a question.
2. **Transcript Retrieval**: Using `youtube-transcript-api`, the transcript is fetched.
3. **Text Processing**: Transcripts are split into chunks with context overlap.
4. **Semantic Indexing**: Chunks are embedded and stored in a **FAISS** vector DB.
5. **Contextual Retrieval**: Most relevant chunks are retrieved using cosine similarity.
6. **Answer Generation**: LLM uses the retrieved context to generate an accurate response.

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/youtube-chatbot-qa.git
   cd youtube-chatbot-qa
