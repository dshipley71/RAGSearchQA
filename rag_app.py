import os
import io
import base64
import configparser
import shutil

from PIL import Image
import streamlit as st
from datetime import datetime
from pathlib import Path
from rag_app_haystack import RAGApplication

# Load settings from settings.ini
config = configparser.ConfigParser()
config.read("settings.ini")

st.set_page_config(
    page_title="RAG-Search-QA",
    page_icon="assets/ai_sidebar.jpg",
    layout="wide"
)

def load_settings():
    """
    Load application settings from settings.ini
    """
    settings = {
        "log_level": config.get("Settings", "LogLevel", fallback="INFO"),
        "device": config.get("Settings", "device", fallback="cuda"),
        "collection_name": config.get("Database", "collection_name", fallback="collections"),
        "persist_path": config.get("Database", "persist_path", fallback=""),
        "embedding_model": config.get("Models", "embedding_model", fallback="models/multilingual-e5-large"),
        "llm_model": config.get("Models", "llm_model", fallback="models/Llama-3.2-3B-Instruct"),
        "split_by": config.get("Documents", "split_by", fallback="sentence"),
        "split_length": config.getint("Documents", "split_length", fallback=150),
        "split_overlap": config.getint("Documents", "split_overlap", fallback=50),
        "split_threshold": config.getint("Documents", "split_threshold", fallback=10),
        "policy": config.get("Documents", "policy", fallback="overwrite"),
        "tika_url": config.get("Dcouments", "tika_url", fallback="http://localhost:9998/tika"),
        "unstructured_url": config.get("Dcouments", "unstructured_url", fallback="http://localhost:8000/general/v0/general"),
        "remove_empty_lines": config.getboolean("Documents", "remove_empty_lines", fallback=True),
        "remove_extra_whitespaces": config.getboolean("Documents", "remove_extra_whitespaces", fallback=True),
        "remove_repeated_substrings": config.getboolean("Documents", "remove_repeated_substrings", fallback=True),
        "max_new_tokens": config.getint("LLM", "max_new_tokens", fallback=500),
        "temperature": config.getfloat("LLM", "temperature", fallback=0.1),
        "top_k": config.getint("LLM", "top_k", fallback=5),
        "top_p": config.getfloat("LLM", "top_p", fallback=0.95),
        "repetition_penalty": config.getfloat("LLM", "repetition_penalty", fallback=1.15),
        "return_full_text": config.getboolean("LLM", "return_full_text", fallback=False),
        "template": config.get("Prompts", "template", fallback=""),
        "pipeline": config.get("Indexing Pipeline", "pipeline", fallback="tika"),
    }
    settings["persist_path"] = settings["persist_path"].strip() or None
    settings["policy"] = settings["policy"] if settings["policy"] in {"overwrite", "skip", "fail"} else None
    return settings

settings = load_settings()

# Set up temporary directory
TEMP_UPLOAD_DIR = "./temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def initialize_rag_application(system_prompt):
    """
    Initialize the RAGApplication using settings.
    """
    rag = RAGApplication(
        template=system_prompt, #settings["template"],
        collection_name=settings["collection_name"],
        persist_path=settings["persist_path"],
        embedding_model=settings["embedding_model"],
        llm_model=settings["llm_model"],
        remove_empty_lines=settings["remove_empty_lines"],
        remove_extra_whitespaces=settings["remove_extra_whitespaces"],
        remove_repeated_substrings=settings["remove_repeated_substrings"],
        split_by=settings["split_by"],
        split_length=settings["split_length"],
        split_overlap=settings["split_overlap"],
        split_threshold=settings["split_threshold"],
        policy=settings["policy"],
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        top_k=settings["top_k"],
        repetition_penalty=settings["repetition_penalty"],
        return_full_text=settings["return_full_text"],
        tika_url=settings["tika_url"],
        unstructured_url=settings["unstructured_url"],
    )
    rag.create_document_store()
    return rag

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory."""
    file_paths = []
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        file_paths.append(Path(temp_file_path))
    return file_paths

def main():
    """Main function to run the Streamlit application."""

    # Custom CSS
    st.markdown(
        '''
        <style>
        div[data-testid="column"] { margin: 1rem; }
        div.stApp > main > div.block-container {
            display: flex; justify-content: center; align-items: flex-start; gap: 2rem;
        }
        div[data-testid="column"] { width: 100% !important; }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Sidebar: logo image
    st.sidebar.image("assets/ai_sidebar.jpg")

    # Sidebar: System prompt
    st.sidebar.title("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Customize your prompt:",
        value=settings["template"],
        height=220,
        # disabled=True
    )

    # Sidebar: Upload Documents
    st.sidebar.title("Upload Documents")
    data_files = st.sidebar.file_uploader("Upload your files:", accept_multiple_files=True)

    # Main content
    st.image("assets/ai_banner.jpg", use_container_width=True)
    st.header("AI Document Q&A")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    rag = initialize_rag_application(system_prompt)

    if st.sidebar.button("Submit & Process"):
        if not data_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Indexing documents..."):
                file_paths = save_uploaded_files(data_files)
                if settings["pipeline"] == "tika":
                    rag.tika_extractor(file_paths)
                elif settings["pipeline"] == "unstructured":
                    rag.unstructured_extractor(file_paths)
                elif settings["pipeline"] == "haystack":
                    rag.haystack_extractor(file_paths)
                else:
                    st.sidebar.error(f"Unsupported extraction pipeline: {settings['pipeline']}.")
                st.sidebar.success("Documents indexed successfully!")

                # remove temp directory
                for file in file_paths:
                    os.remove(file)

    user_question = st.text_input("Ask a Question from the Files")
    if user_question:
        with st.spinner("Generating answer..."):
            try:
                result_dict = rag.run_rag(question=user_question)
                final_answer = result_dict.get("Answer", "")
                retrieved_sources = result_dict.get("Source", [])

                st.session_state.conversation.append({
                    "question": user_question,
                    "answer": final_answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source_documents": retrieved_sources,
                })

                st.write("**Reply:**", final_answer)
                st.write("### Source Documents")
                for src in retrieved_sources:
                    st.write(
                        f"**Path:** {os.path.basename(src.get('file_path', 'N/A'))} | "
                        f"**Page:** {src.get('page_number', 'N/A')} | "
                        f"**Content:** {src.get('content', 'N/A')} | "
                        f"**Score:** {float(src.get('score', 'N/A')):.5f}"
                    )
            except Exception as e:
                st.error(f"Error during retrieval or generation: {str(e)}")

    if st.session_state.conversation:
        st.write("### Conversation History")
        for entry in st.session_state.conversation[::-1]:
            with st.expander(f"Q: {entry['question']} ({entry['timestamp']})"):
                st.write(f"**A:** {entry['answer']}")

if __name__ == "__main__":
    main()
