import os
import io
import base64
from PIL import Image
import streamlit as st
from datetime import datetime
from pathlib import Path
from rag_app_haystack import RAGApplication
from pprint import pprint

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["SCARF_NO_ANALYTICS"] = "true"

# Location of NLTK data (if you need it)
os.environ["NLTK_DATA"] = "./nltk_data"

# Create a temporary directory for uploaded files (if it doesn't already exist)
TEMP_UPLOAD_DIR = "./temp_uploads"
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


@st.cache_resource
def initialize_rag_application(
    template,
    persist_path,
    embedding_model,
    llm_model,
    max_new_tokens,
    temperature,
    top_p,
    chunk_size,
    chunk_overlap,
    use_chunking
):
    """
    Initializes and caches the RAGApplication object from rag_app_haystack.
    If persist_path is None or empty, we get a non-persistent DB.
    If persist_path is provided, we get a persistent DB.
    """
    # Map your Streamlit UI settings to the RAGApplication constructor
    rag = RAGApplication(
        template=template,
        collection_name="my_collection",
        persist_path=persist_path,             # None for ephemeral DB
        embedding_model=embedding_model,
        llm_model=llm_model,
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True,
        split_by="sentence",                   # This can be "sentence", "word", etc.
        split_length=chunk_size if use_chunking else 150,
        split_overlap=chunk_overlap if use_chunking else 50,
        split_threshold=10,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=5,                               # You can expose this in the UI if needed
        repetition_penalty=1.15,
        return_full_text=False,                # Whether to return model's entire text
    )
    rag.create_document_store()
    return rag


def save_uploaded_files(uploaded_files):
    """
    Saves files uploaded via Streamlit to a local temp folder so that we can pass them
    to `RAGApplication.run_embedder()`. Returns a list of file paths (Path objects).
    """
    file_paths = []
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        file_paths.append(Path(temp_file_path))
    return file_paths


# Configure the page layout
st.set_page_config(
    page_title="RAG-Search-QA",
    page_icon="assets/ai_sidebar.jpg",
    layout="wide"
)


# Custom CSS for styling
custom_style = '''
    <style>
    div[data-testid="column"] {
        margin: 1rem;
    }
    div.stApp > main > div.block-container {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 2rem;
    }
    div[data-testid="column"] {
        width: 100% !important;
    }
    div[data-testid="column"] > div:first-child {
        flex: 1;
    }
    .stSidebar {
        margin-left: 1rem;
    }
    .custom-settings-title {
        font-size: 24px; /* Match the font size of the sidebar titles */
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
'''
st.markdown(custom_style, unsafe_allow_html=True)


def main():
    """
    Main function to run the Streamlit application for chatting with documents
    using the RAGApplication from rag_app_haystack.py.
    """

    # Sidebar: Image
    st.sidebar.image("assets/ai_sidebar.jpg")

    # A default prompt template that gets passed to RAGApplication
    default_prompt = """You are a helpful assistant. Answer the question as detailed as possible from
the provided context and conversation history. If the answer is not in the provided context, just say,
'answer is not available in the context'.

Conversation history:
{% for memory in memories %}
    {{ memory.content }}
{% endfor %}

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question:
{{ question }}

Answer:
"""

    # Sidebar: System Prompt
    st.sidebar.title("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Customize your prompt:",
        value=default_prompt,
        height=150,
        disabled=True
    )

    # Sidebar: File Upload
    st.sidebar.title("Upload Documents")
    data_files = st.sidebar.file_uploader("Upload your files:", accept_multiple_files=True)

    # Main Content Layout
    col2, col3 = st.columns([3, 1])

    # Define variables in col3 for global access
    with col3:
        st.markdown('<div class="custom-settings-title">Settings</div>', unsafe_allow_html=True)

        # Database Selection (Only ChromaDB is used)
        with st.expander(label="Database Path:", expanded=False):
            # st.write("Using ChromaDB (non-persistent by default).")
            # st.write("If you provide a path, the database will be persistent.")
            vector_store_path = st.text_input(
                "ChromaDB Persist Path:",
                "",
                help="Leave blank or set to None for an in-memory database. "
                     "If you specify a path, Chroma will persist data there."
            )
            # Normalize input (strip spaces); if empty => None
            persist_path = vector_store_path.strip() if vector_store_path.strip() != "" else None

        # Model Selection
        with st.expander(label="Model Selection:", expanded=False):
            embedding_model_name = st.selectbox(
                "Embedding Model",
                ["multilingual-e5-large", "all-MiniLM-L6-v2"],
                help="Default set to multilingual-e5-large. If performance is slow, try smaller models."
            )
            llm_model = st.selectbox(
                "Large Language Model",
                ["Llama-3.2-3B-Instruct"],
                help="Default set to Llama-3.2-3B-Instruct"
            )

            # Prepend local folder paths to replicate structure in rag_app_haystack
            embedding_model_name = f"models/{embedding_model_name}"
            llm_model = f"models/{llm_model}"

        # Document Settings
        with st.expander(label="Document Settings:", expanded=False):
            chunk_text = st.checkbox("Chunk Text", value=True)
            chunk_size = st.number_input("Chunk Size", min_value=50, max_value=2000, value=150, step=50)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

        # LLM Settings
        with st.expander(label="LLM Settings:", expanded=False):
            max_new_tokens = st.number_input("Max New Tokens", min_value=50, max_value=65536, value=500, step=50)
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
            top_p = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    # Main content in col2
    with col2:
        st.image("assets/ai_banner.jpg", use_container_width=True)
        st.header("AI Document Q&A")

        # Initialize session state for conversation history
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        # Build our RAG pipeline only once or whenever settings change
        # The @st.cache_resource above handles re-instantiating on settings changes
        rag = initialize_rag_application(
            template=system_prompt,
            persist_path=persist_path,
            embedding_model=embedding_model_name,
            llm_model=llm_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_chunking=chunk_text,
        )

        # Process new uploads on user’s request
        if st.sidebar.button("Submit & Process"):
            if not data_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Indexing documents..."):
                    file_paths = save_uploaded_files(data_files)
                    rag.run_embedder(file_paths)
                st.sidebar.success("Documents indexed successfully!")

        # Q&A section
        user_question = st.text_input("Ask a Question from the Files")
        if user_question:
            with st.spinner("Generating answer..."):
                try:
                    # Basic RAG retrieval & generation (no conversation memory injection here):
                    result_dict = rag.run_rag(question=user_question)
                    # st.write(result_dict)

                    # The RAGApplication returns a dict with "Answer", "Query", "Source"
                    final_answer = result_dict.get("Answer", "")
                    retrieved_sources = result_dict.get("Source", [])

                    # Update the conversation history (in local Streamlit state)
                    st.session_state.conversation.append({
                        "question": user_question,
                        "answer": final_answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_documents": retrieved_sources,
                    })

                    st.write("**Reply:**", final_answer)
                    st.write(f"**LLM Model:** {llm_model}")
                    st.write("### Source Documents")
                    if isinstance(retrieved_sources, list):
                        for src in retrieved_sources:
                            st.write(
                                f"**Path:** {src.get('file_path', 'N/A')} | "
                                f"**Page:** {src.get('page_number', 'N/A')} | "
                                f"**Score:** {float(src.get('score', 'N/A')):.5f}"
                            )
                    else:
                        st.write("No source documents found.")

                except Exception as e:
                    st.error(f"Error during retrieval or generation: {str(e)}")

        # Display conversation history
        if st.session_state.conversation:
            st.write("### Conversation History")
            # Reverse to show newest first
            for entry in st.session_state.conversation[::-1]:
                with st.expander(f"Q: {entry['question']} ({entry['timestamp']})"):
                    st.write(f"**A:** {entry['answer']}")
                    st.write("#### Source Documents")
                    for src in entry["source_documents"]:
                        try:
                            st.write(
                                f"**Path:** {src.get('file_path', 'N/A')} | "
                                f"**Page:** {src.get('page_number', 'N/A')} | "
                                f"**Score:** {float(src.get('score', 'N/A')):.5f}"
                            )
                        except:
                            st.write("No source documents found.")
                            break

            # Download button for conversation history
            download_button_style = """
            <style>
            .download-button {
                background-color: white;
                color: black;
                font-size: 16px;
                font-weight: normal;
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid #d3d3d3;
                cursor: pointer;
                text-align: center;
                max-width: 255px;
                margin: 0;
                display: block;
                text-decoration: none;
            }
            .download-button:hover {
                background-color: #dcdcdc;
                text-decoration: none;
            }
            </style>
            """
            st.markdown(download_button_style, unsafe_allow_html=True)

            conversation_str = "\n\n".join(
                [
                    f"Q: {entry['question']}\nA: {entry['answer']}\nTimestamp: {entry['timestamp']}"
                    for entry in st.session_state.conversation
                ]
            )
            conversation_bytes = conversation_str.encode("utf-8")

            st.markdown(
                f"""
                <a href="data:text/plain;base64,{base64.b64encode(conversation_bytes).decode()}"
                download="conversation_history.txt" class="download-button">
                    Download Conversation History
                </a>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
