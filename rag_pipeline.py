import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq
import os
from dotenv import load_dotenv
 
load_dotenv()
 
def extract_text_with_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages
 
def split_text_with_pages(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
    docs = []
    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"page": page_data["page"]}
            ))
    return docs
 
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
 
def generate_answer(context, question):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""You are a precise document analyst. Answer the question based ONLY on the context below.
 
IMPORTANT RULES:
- Be thorough and list ALL relevant items — do not skip or summarize
- If counting items (like certifications, skills, etc.), list every single one you find
- If the answer is not in the context, say "I couldn't find this in the document."
- Do not add information from outside the context
 
Context:
{context}
 
Question: {question}
 
Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content