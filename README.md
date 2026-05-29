# 📖 DocuMind — AI Document Q&A

Upload any PDF and ask questions about it using AI.

## Features
- Multi-PDF support
- Chat history
- Page-level source references
- Download answers as text files

## Tech Stack
- Streamlit · LangChain · FAISS · HuggingFace · Groq (LLaMA 3.3 70B)

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to a `.env` file: `GROQ_API_KEY=your_key`
4. Run: `streamlit run app.py`
