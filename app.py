import streamlit as st
from rag_pipeline import extract_text_with_pages, split_text_with_pages, create_vectorstore, generate_answer
import tempfile
import os
from datetime import datetime
 
st.set_page_config(page_title="DocuMind", page_icon="📖", layout="wide")
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono&display=swap');
 
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #f7f4ef !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 820px !important; }
 
section[data-testid="stFileUploadDropzone"] {
    background: white !important;
    border: 2px dashed #d4c9f0 !important;
    border-radius: 16px !important;
    transition: all 0.3s !important;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #7c5cbf !important;
    background: #faf8ff !important;
}
 
.stTextInput > div > div > input {
    background: white !important;
    border: 2px solid #e8e2f5 !important;
    border-radius: 14px !important;
    color: #1a1a1a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: all 0.3s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c5cbf !important;
    box-shadow: 0 0 0 4px rgba(124,92,191,0.12) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #bbb !important; }
.stTextInput label, .stFileUploader label { display: none !important; }
 
.stButton > button {
    background: linear-gradient(135deg, #7c5cbf, #a078d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(124,92,191,0.3) !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,92,191,0.45) !important;
}
.stDownloadButton > button {
    background: white !important;
    color: #7c5cbf !important;
    border: 2px solid #d8ccf5 !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    width: auto !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
    margin-top: 0 !important;
}
.stDownloadButton > button:hover {
    background: #f0ebff !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)
 
# NAV
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding:1rem 0 2.5rem;">
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#7c5cbf,#c084fc);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📖</div>
        <span style="font-weight:800;font-size:1.15rem;color:#1a1a1a;letter-spacing:-0.02em;">Docu<span style="color:#7c5cbf;">Mind</span></span>
    </div>
    <div style="display:flex;gap:8px;">
        <span style="font-size:0.72rem;color:#7c5cbf;font-family:'DM Mono',monospace;background:#f0ebff;padding:5px 12px;border-radius:20px;border:1px solid #d8ccf5;">LLaMA 3.3 70B</span>
        <span style="font-size:0.72rem;color:#7c5cbf;font-family:'DM Mono',monospace;background:#f0ebff;padding:5px 12px;border-radius:20px;border:1px solid #d8ccf5;">FAISS · RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
# HERO
st.markdown("""
<div style="margin-bottom:2.5rem;">
    <div style="display:inline-flex;align-items:center;gap:8px;background:#f0ebff;border:1px solid #d8ccf5;border-radius:30px;padding:5px 14px;margin-bottom:1.25rem;">
        <div style="width:6px;height:6px;background:#7c5cbf;border-radius:50%;"></div>
        <span style="font-size:0.72rem;color:#7c5cbf;font-weight:600;letter-spacing:0.05em;font-family:'DM Mono',monospace;">AI-POWERED DOCUMENT ANALYSIS</span>
    </div>
    <h1 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:2.8rem;font-weight:800;color:#1a1a1a;letter-spacing:-0.04em;line-height:1.1;margin:0 0 1rem;">Ask your documents <span style="color:#7c5cbf;">anything.</span></h1>
    <p style="font-size:1rem;color:#777;line-height:1.7;margin:0 0 2rem;max-width:500px;">Upload multiple PDFs and get instant, precise answers with page references and full chat history.</p>
    <div style="display:flex;gap:2.5rem;">
        <div><div style="font-size:1.4rem;font-weight:800;color:#7c5cbf;">Multi</div><div style="font-size:0.68rem;color:#aaa;font-family:'DM Mono',monospace;margin-top:2px;">PDF SUPPORT</div></div>
        <div style="width:1px;background:#ede8f8;"></div>
        <div><div style="font-size:1.4rem;font-weight:800;color:#7c5cbf;">6x</div><div style="font-size:0.68rem;color:#aaa;font-family:'DM Mono',monospace;margin-top:2px;">PASSAGES SCANNED</div></div>
        <div style="width:1px;background:#ede8f8;"></div>
        <div><div style="font-size:1.4rem;font-weight:800;color:#7c5cbf;">70B</div><div style="font-size:0.68rem;color:#aaa;font-family:'DM Mono',monospace;margin-top:2px;">MODEL PARAMS</div></div>
        <div style="width:1px;background:#ede8f8;"></div>
        <div><div style="font-size:1.4rem;font-weight:800;color:#7c5cbf;">Page</div><div style="font-size:0.68rem;color:#aaa;font-family:'DM Mono',monospace;margin-top:2px;">REFERENCES</div></div>
    </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#d8ccf5,transparent);margin-bottom:2.5rem;"></div>
""", unsafe_allow_html=True)
 
# UPLOAD CARD
st.markdown("""
<div style="background:white;border-radius:20px;border:1px solid #ede8f8;box-shadow:0 2px 20px rgba(124,92,191,0.07);padding:1.75rem;margin-bottom:1.25rem;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.25rem;">
        <div style="width:26px;height:26px;background:#f0ebff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;">📁</div>
        <span style="font-weight:700;font-size:0.88rem;color:#1a1a1a;">Upload PDFs</span>
        <span style="font-size:0.7rem;color:#aaa;font-family:'DM Mono',monospace;margin-left:4px;">MULTIPLE FILES SUPPORTED</span>
    </div>
""", unsafe_allow_html=True)
 
uploaded_files = st.file_uploader("upload", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
 
st.markdown("</div>", unsafe_allow_html=True)
 
if uploaded_files:
    new_names = sorted([f.name for f in uploaded_files])
    if new_names != st.session_state.uploaded_names:
        with st.spinner(f"Indexing {len(uploaded_files)} document(s)..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                pages = extract_text_with_pages(tmp_path)
                docs = split_text_with_pages(pages)
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                all_docs.extend(docs)
                os.unlink(tmp_path)
            st.session_state.vectorstore = create_vectorstore(all_docs)
            st.session_state.uploaded_names = new_names
            st.session_state.total_chunks = len(all_docs)
            st.session_state.chat_history = []
 
    names_html = "".join([
        f'<span style="font-size:0.72rem;color:#7c5cbf;font-family:DM Mono,monospace;background:#f0ebff;padding:3px 10px;border-radius:20px;border:1px solid #d8ccf5;">{n}</span>'
        for n in st.session_state.uploaded_names
    ])
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#f0ebff,#fdf4ff);border:1px solid #d8ccf5;border-radius:14px;padding:1rem 1.25rem;margin-bottom:1.25rem;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">{names_html}</div>
    <p style="font-size:0.7rem;color:#7c5cbf;margin:0;font-family:'DM Mono',monospace;">{st.session_state.total_chunks} CHUNKS INDEXED · {len(st.session_state.uploaded_names)} FILE(S) · READY</p>
</div>
""", unsafe_allow_html=True)
 
    # CHAT HISTORY
    if st.session_state.chat_history:
        st.markdown("""
<div style="background:white;border-radius:20px;border:1px solid #ede8f8;box-shadow:0 2px 20px rgba(124,92,191,0.07);padding:1.75rem;margin-bottom:1.25rem;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.5rem;">
        <div style="width:26px;height:26px;background:#f0ebff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;">🕓</div>
        <span style="font-weight:700;font-size:0.88rem;color:#1a1a1a;">Chat History</span>
    </div>
""", unsafe_allow_html=True)
 
        for i, entry in enumerate(st.session_state.chat_history):
            export_text = f"Q: {entry['question']}\n\nA: {entry['answer']}\n\nSources:\n"
            for s in entry['sources']:
                export_text += f"- {s['source']} (Page {s['page']}): {s['text']}\n"
 
            st.markdown(f"""
<div style="margin-bottom:1.25rem;padding-bottom:1.25rem;border-bottom:1px solid #f0ebff;">
    <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:0.75rem;">
        <div style="width:26px;height:26px;background:#f0ebff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.75rem;flex-shrink:0;margin-top:2px;">🙋</div>
        <div style="background:#f7f4ef;border-radius:0 12px 12px 12px;padding:0.75rem 1rem;flex:1;">
            <p style="font-size:0.9rem;color:#1a1a1a;font-weight:600;margin:0;">{entry['question']}</p>
            <p style="font-size:0.68rem;color:#bbb;font-family:'DM Mono',monospace;margin:4px 0 0;">{entry['time']}</p>
        </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:10px;">
        <div style="width:26px;height:26px;background:linear-gradient(135deg,#7c5cbf,#c084fc);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.75rem;flex-shrink:0;margin-top:2px;">✨</div>
        <div style="background:#faf8ff;border:1px solid #ede8f8;border-left:3px solid #7c5cbf;border-radius:0 12px 12px 12px;padding:0.75rem 1rem;flex:1;">
            <p style="font-size:0.9rem;color:#1a1a1a;line-height:1.75;margin:0 0 0.75rem;white-space:pre-wrap;">{entry['answer']}</p>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:0.5rem;">
                {"".join([f'<span style="font-size:0.68rem;color:#7c5cbf;background:#f0ebff;padding:3px 8px;border-radius:6px;font-family:DM Mono,monospace;border:1px solid #d8ccf5;">📄 {s["source"]} · p.{s["page"]}</span>' for s in entry["sources"]])}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
 
            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button(
                    label="⬇ Export",
                    data=export_text,
                    file_name=f"answer_{i+1}.txt",
                    mime="text/plain",
                    key=f"dl_{i}"
                )
 
        st.markdown("</div>", unsafe_allow_html=True)
 
        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
 
    # QUESTION CARD
    st.markdown("""
<div style="background:white;border-radius:20px;border:1px solid #ede8f8;box-shadow:0 2px 20px rgba(124,92,191,0.07);padding:1.75rem;margin-bottom:1.25rem;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.25rem;">
        <div style="width:26px;height:26px;background:#f0ebff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:0.85rem;">💬</div>
        <span style="font-weight:700;font-size:0.88rem;color:#1a1a1a;">Ask a question</span>
    </div>
""", unsafe_allow_html=True)
 
    question = st.text_input("q", placeholder="e.g. What certifications are mentioned?", label_visibility="collapsed")
    ask = st.button("Get Answer →")
    st.markdown("</div>", unsafe_allow_html=True)
 
    if ask and question:
        with st.spinner("Analysing document..."):
            context_docs = st.session_state.vectorstore.similarity_search(question, k=6)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            answer = generate_answer(context, question)
 
            sources = []
            seen = set()
            for doc in context_docs:
                key = (doc.metadata.get("source",""), doc.metadata.get("page", "?"))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source": doc.metadata.get("source", "Document"),
                        "page": doc.metadata.get("page", "?"),
                        "text": doc.page_content[:120] + "..."
                    })
 
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "time": datetime.now().strftime("%H:%M · %d %b %Y")
            })
            st.rerun()
 
else:
    st.markdown("""
<div style="background:white;border-radius:20px;border:2px dashed #d8ccf5;padding:3rem 2rem;text-align:center;margin-bottom:1.25rem;">
    <div style="font-size:2.5rem;margin-bottom:1rem;">📂</div>
    <p style="font-weight:700;font-size:1rem;color:#1a1a1a;margin:0 0 0.4rem;">No documents uploaded yet</p>
    <p style="font-size:0.85rem;color:#aaa;margin:0;">Upload one or more PDFs above to get started</p>
</div>
""", unsafe_allow_html=True)