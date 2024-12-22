import os
import io
import base64
from PIL import Image
import streamlit as st
from datetime import datetime
from rag_application import RAGApplication
from pprint import pprint

# disable telemetry
os.environ["ANONYMIZED_TELEMETRY"]="False"
os.environ["SCARF_NO_ANALYTICS"]="true"

# location of NLTK data
os.environ["NLTK_DATA"]="./nltk_data"

@st.cache_resource
def initialize_rag_application(vector_database, vector_store_path, embedding_model_name, temperature, max_new_tokens, top_p):
    """
    Initializes and caches the RAGApplication object.
    """
    return RAGApplication(
        embedding_model_name=embedding_model_name,
        vector_database=vector_database,
        vector_store_path=vector_store_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p
    )

# Configure the page layout
st.set_page_config(
    page_title="RAG-Search-QA",
    page_icon="assets/ai_sidebar.jpg",
    layout="wide"
)

# Custom CSS for styling columns with margins
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
    /* Adjust font size for the "Settings" title */
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
    Main function to run the Streamlit application for chatting with files using RAG and Llama-3.2.
    """
    # Sidebar: Image
    st.sidebar.image("assets/ai_sidebar.jpg")

    prompt = """ You are a helpful assistant. Answer the question as detailed as possible from
the provided context and conversation history. If the answer is not in the provided context, just say,
'answer is not available in the context'.
Context: {context}
Question: {question}
"""

    # Sidebar: System Prompt
    st.sidebar.title("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Customize your prompt:",
        # value="You are a helpful assistant. Answer the question as detailed as possible from the provided context. "
        #       "If the answer is not in the provided context, just say, 'answer is not available in the context'.",
        value=prompt,
        height=150
    )

    # Sidebar: File Upload
    st.sidebar.title("Upload Documents")
    data_files = st.sidebar.file_uploader("Upload your files:", accept_multiple_files=True)

    # Main Content Layout
    col2, col3 = st.columns([3, 1])

    # Define variables in col3 for global access
    with col3:
        # Custom title with adjusted font size
        st.markdown('<div class="custom-settings-title">Settings</div>', unsafe_allow_html=True)

        # Database Selection
        with st.expander(label="Database Selection:", expanded=False):
            vector_database = st.selectbox("Database:", ["ChromaDB", "FAISS"], help="Default database set to FAISS")
            vector_store_path = st.text_input("Vector Store Path", "data/vectorstore/my_store", help="Path to vector database storage used only with ChromaDB.")

        # Model Selection
        with st.expander(label="Model Selection:", expanded=False):
            embedding_model_name = st.selectbox("Embedding Model", ["multilingual-e5-large", "all-MiniLM-L6-v2"], help="Default set to multilingual-e5-large. If performance is slow for single GPU usage, use all-MiniLM-L6-v2.")
            # vlm_model = st.selectbox("Vision Language Model:", ["Llama-3.2-11B-Vision-Instruct"], help="Default set to Llama-3.2-11B-Vision-Instruct")
            llm_model = st.selectbox("Large Language Model", ["Llama-3.2-3B-Instruct"], help="Default set to Llama-3.2-3B-Instruct")
            
            embedding_model_name = "models/" + embedding_model_name
            # vlm_model = "models/" + vlm_model
            llm_model = "models/" + llm_model

        # Document Settings
        with st.expander(label="Document Settings:", expanded=False):
            chunk_text = st.checkbox("Chunk Text", value=True)
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1024, step=100)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=int(chunk_size/10), step=50)
            num_docs = st.number_input("Number of Documents to Retrieve", min_value=1, max_value=10, value=5, step=1)

        # LLM Settings
        with st.expander(label="LLM Settings:", expanded=False):
            max_new_tokens = st.number_input("Max New Tokens", min_value=50, max_value=65536, value=8192, step=50)
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
            top_p = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    # Main Content Area
    with col2:
        # image = Image.open("assets/ai_banner.jpg")
        # buffered = io.BytesIO()
        # image.save(buffered, format="JPEG")
        # img_str = base64.b64encode(buffered.getvalue()).decode()
        # html_temp = f"""
        # <div style="text-align: center;">
        # <img src="data:image/jpeg.base64,{img_str}"/>
        # </div>
        # """
        st.image("assets/ai_banner.jpg", use_container_width=True)        
        st.header("AI Document Q&A")

        # Initialize session state for conversation history
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        user_question = st.text_input("Ask a Question from the Files")
        if user_question:
            try:
                # Initialize RAG application
                rag = initialize_rag_application(
                    vector_database,
                    vector_store_path,
                    embedding_model_name,
                    temperature,
                    max_new_tokens,
                    top_p
                )

                # Load vector store if not already loaded
                if not hasattr(st.session_state, "vector_store"):
                    st.session_state.vector_store = rag.load_vector_store()

                # TODO update system prompt to include conversation history

                response = rag.get_conversational_chain(
                    user_question, llm_model, system_prompt
                )

                # Process response
                # if response and "result" in response:
                if response and "answer" in response:
                    result = response["answer"]
                    source_documents = response.get("context", [])

                    # Update conversation history
                    st.session_state.conversation.append({
                        "question": user_question,
                        "answer": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_documents": source_documents
                    })

                    # Display response
                    st.write("Reply: ", result)
                    st.write(f"**LLM Model:** {llm_model}")
                    st.write(f"**Vector Database:** {vector_database}")
                    st.write("### Source Documents")
                    for doc in source_documents:
                        metadata = doc.metadata
                        st.write(f"**Source:** {metadata.get('source', 'Unknown')} | "
                                 f"**Page Number:** {metadata.get('page_number', 'N/A')} | "
                                 f"**File Type:** {metadata.get('filetype', 'N/A')}")

                else:
                    st.error("No result found. Please try again.")

            except Exception as e:
                st.error(f"An error occurred while retrieving the response: {str(e)}")

        # Display conversation history only after a question is asked
        if st.session_state.conversation:
            st.write("### Conversation History")
            for entry in st.session_state.conversation[::-1]:  # Reverse to show latest first
                with st.expander(f"Q: {entry['question']} ({entry['timestamp']})"):
                    st.write(f"**A:** {entry['answer']}")
                    st.write("### Source Documents")
                    for doc in entry["source_documents"]:
                        metadata = doc.metadata
                        st.write(f"**Source:** {metadata.get('source', 'Unknown')} | "
                                 f"**Page Number:** {metadata.get('page_number', 'N/A')} | "
                                 f"**File Type:** {metadata.get('filetype', 'N/A')}")

            # # Add a download button for conversation history
            # if st.session_state.conversation:
            #     conversation_str = "\n\n".join(
            #         [f"Q: {entry['question']}\nA: {entry['answer']}\nTimestamp: {entry['timestamp']}" for entry in st.session_state.conversation]
            #     )
            #     conversation_bytes = conversation_str.encode("utf-8")
            #     st.download_button(
            #         label="Download Conversation History",
            #         data=conversation_bytes,
            #         file_name="conversation_history.txt",
            #         mime="text/plain"
            #     )

            # Custom CSS for the download button
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
                    text-decoration: none; /* removes underline */
                }
                .download-button:hover {
                    background-color: #dcdcdc;
                    text-decoration: none; /* prevent underline on hover */
                }
                </style>
            """
            st.markdown(download_button_style, unsafe_allow_html=True)

            # Add a download button for conversation history
            if st.session_state.conversation:
                conversation_str = "\n\n".join(
                    [f"Q: {entry['question']}\nA: {entry['answer']}\nTimestamp: {entry['timestamp']}" for entry in st.session_state.conversation]
                )
                conversation_bytes = conversation_str.encode("utf-8")
                st.markdown(
                    f"""
                    <a href="data:text/plain;base64,{base64.b64encode(conversation_bytes).decode()}" download="conversation_history.txt" class="download-button">
                        Download Conversation History
                    </a>
                    """,
                    unsafe_allow_html=True
                )

    # Process documents on submission
    if st.sidebar.button("Submit & Process"):
        try:
            # Initialize RAG application
            rag = initialize_rag_application(
                vector_database,
                vector_store_path,
                embedding_model_name,
                temperature,
                max_new_tokens,
                top_p
            )

            if not data_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing..."):
                    raw_documents = rag.read_data(data_files)
                    if not raw_documents:
                        st.error("No valid documents were processed. Please check your files.")
                    else:
                        text_chunks = (
                            rag.get_chunks(raw_documents, chunk_size, chunk_overlap) if chunk_text else raw_documents
                        )
                        # rag.store_vector_data(text_chunks)
                        rag.store_vector_data(text_chunks)
                        st.sidebar.success("Documents processed and stored successfully.")

        except Exception as e:
            st.sidebar.error(f"An error occurred while processing the files: {str(e)}")

if __name__ == "__main__":
    main()