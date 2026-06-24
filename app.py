import os
import re
import glob
import requests
import streamlit as st
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Set page configuration with a premium look
st.set_page_config(
    page_title="Veda & Scripture Q&A Explorer",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and animations
st.markdown("""
<style>
    /* Gradient Title */
    .title-gradient {
        background: linear-gradient(90deg, #FF8C00, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Subtle subtitle styling */
    .subtitle {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sidebar headers */
    .sidebar-header {
        color: #FF8C00;
        font-weight: 700;
        font-size: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Stats display box */
    .stats-box {
        background-color: rgba(255, 140, 0, 0.1);
        border: 1px solid rgba(255, 140, 0, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Load book context (full text stats)
@st.cache_data
def get_book_statistics(directory):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    txt_files = [f for f in txt_files if "combined_book" not in os.path.basename(f)]
    
    if not txt_files:
        return 0, 0, 0
        
    total_words = 0
    total_chars = 0
    for filepath in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                total_words += len(content.split())
                total_chars += len(content)
        except Exception:
            continue
            
    return len(txt_files), total_words, total_chars


# Local keyword-based text retriever (RAG)
def retrieve_relevant_pages(query, directory, top_n=5):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    txt_files = [f for f in txt_files if "combined_book" not in os.path.basename(f)]
    
    pages = []
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        name_without_ext, _ = os.path.splitext(filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            pages.append({"name": name_without_ext, "text": text})
        except Exception:
            continue
            
    if not pages:
        return "", []
        
    # Keyword extraction from query
    words = re.findall(r'\w+', query.lower())
    stop_words = {
        "what", "is", "the", "of", "and", "in", "to", "a", "for", "on", "with", "this", "about", 
        "page", "book", "answer", "question", "है", "का", "की", "के", "में", "पर", "और", "को", "तो"
    }
    keywords = [w for w in words if len(w) > 1 and w not in stop_words]
    
    # If no keywords, fallback to first pages
    if not keywords:
        fallback_pages = pages[:top_n]
        context = "\n\n".join([f"--- PAGE: {p['name']} ---\n{p['text']}" for p in fallback_pages])
        return context, [p['name'] for p in fallback_pages]
        
    # Score each page by keyword frequencies
    scored_pages = []
    for p in pages:
        score = 0
        page_text_lower = p["text"].lower()
        for kw in keywords:
            score += page_text_lower.count(kw)
        scored_pages.append((score, p))
        
    # Sort by score descending
    scored_pages.sort(key=lambda x: x[0], reverse=True)
    
    # Take top_n pages with matches
    selected_pages = [sp[1] for sp in scored_pages[:top_n] if sp[0] > 0]
    
    if not selected_pages:
        selected_pages = pages[:top_n]
        
    selected_pages.sort(key=lambda p: p["name"])
    
    parts = []
    cited_pages = []
    for p in selected_pages:
        parts.append(f"\n--- PAGE: {p['name']} ---\n{p['text']}")
        cited_pages.append(p['name'])
        
    return "\n".join(parts), cited_pages


# Load backend secrets securely (Users will not see these in the UI)
openai_base = os.getenv("OPENAI_API_BASE", "https://router.huggingface.co/v1")
api_key = os.getenv("OPENAI_API_KEY", "").strip()
model_name = os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()


# Streamlit sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">🔍 Retrieval Settings</div>', unsafe_allow_html=True)
    num_pages = st.slider(
        "Pages to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Retrieves only the most relevant pages matching your question to stay within context limits."
    )
    
    st.markdown('<div class="sidebar-header">📚 Book Metadata</div>', unsafe_allow_html=True)
    
    # Load and display statistics
    text_dir = os.path.join("data", "book_1_text")
    pages_count, words, chars = get_book_statistics(text_dir)
    
    if pages_count > 0:
        st.markdown(f"""
        <div class="stats-box">
            <strong>Pages Loaded:</strong> {pages_count}<br>
            <strong>Total Words:</strong> {words:,}<br>
            <strong>Characters:</strong> {chars:,}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("⚠️ No book text files found. Make sure to run the OCR script `convert.py` first to generate text files.")
        
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.8rem; color: #888888;">
        <strong>API Engine Details:</strong><br>
        • Mode: Local RAG (Top {num_pages} pages)<br>
        • Strict Mode: Active (only replies from context)
    </div>
    """, unsafe_allow_html=True)

# Main App Header
st.markdown('<div class="title-gradient">Veda & Scripture Q&A Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask classical questions and receive verified answers strictly based on the loaded Sanskrit/Hindi texts.</div>', unsafe_allow_html=True)

# Validate API Key before proceeding (Secure warning)
if not api_key:
    st.error("🔒 Server Configuration Error: API credentials are missing. Please configure OPENAI_API_KEY in your hosting environment settings.")
    st.stop()

# Define system prompt enforcing strictly context-only responses
system_prompt = (
    "You are a scholar of classical Indian literature and scriptures. "
    "Your objective is to answer the user's questions based ONLY on the provided book context.\n\n"
    "Strict Rules:\n"
    "1. Answer using ONLY the information provided in the Book Context. Do not make up facts or use general knowledge.\n"
    "2. If the answer cannot be found in the provided Book Context, respond exactly with:\n"
    "   'I am sorry, but the provided book content does not contain information to answer this question.'\n"
    "3. Respond in the same language as the user's question (e.g. Hindi, Sanskrit, or English).\n"
    "4. When referencing information from the book, cite the Page ID (e.g. 'Page 1782281831776') if it is mentioned in the text."
)

# Initialize chat messages in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "cited_pages" in msg and msg["cited_pages"]:
            cited_str = ", ".join(msg["cited_pages"])
            st.caption(f"🔍 *Citations: Page(s) {cited_str}*")
            with st.expander("📖 View Original Page Images"):
                tabs = st.tabs(msg["cited_pages"])
                for tab, page_name in zip(tabs, msg["cited_pages"]):
                    with tab:
                        img_path = os.path.join("data", "book_1", f"{page_name}.jpg")
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Page {page_name}", use_column_width=True)
                        else:
                            st.warning(f"Image for page {page_name} not found.")

# Handle user input
if user_query := st.chat_input("Ask a question about the book content (e.g. 'What is described in page 1782281831776?')"):
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Check if book files exist
    if pages_count == 0:
        with st.chat_message("assistant"):
            st.error("Cannot answer because no book content is loaded. Please run the OCR script first.")
        st.stop()
        
    # Retrieve only the most relevant pages
    retrieved_context, cited_pages = retrieve_relevant_pages(user_query, text_dir, top_n=num_pages)
    
    # Query Custom OpenAI API
    with st.chat_message("assistant"):
        cited_str = ", ".join(cited_pages)
        st.caption(f"🔍 *Searched and retrieved pages: {cited_str}*")
        
        with st.spinner("Analyzing text and preparing response..."):
            try:
                # Format context prompt
                full_prompt = (
                    f"Book Context:\n"
                    f"==================================\n"
                    f"{retrieved_context}\n"
                    f"==================================\n\n"
                    f"User Question: {user_query}"
                )
                
                # Query via OpenAI-compatible endpoint using requests
                endpoint_url = f"{openai_base.rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1
                }
                
                response = requests.post(endpoint_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    
                    # Display response
                    st.write(response_text)
                    
                    # Display page images
                    if cited_pages:
                        with st.expander("📖 View Original Page Images"):
                            tabs = st.tabs(cited_pages)
                            for tab, page_name in zip(tabs, cited_pages):
                                with tab:
                                    img_path = os.path.join("data", "book_1", f"{page_name}.jpg")
                                    if os.path.exists(img_path):
                                        st.image(img_path, caption=f"Page {page_name}", use_column_width=True)
                                    else:
                                        st.warning(f"Image for page {page_name} not found.")
                                        
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "cited_pages": cited_pages
                    })
                else:
                    st.error(f"API Endpoint returned HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to query model: {e}")
                st.info("Check if your API Key and endpoint in the sidebar are correct.")
