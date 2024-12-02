import os
import streamlit as st
from datetime import datetime
from rag_application import RAGApplication

@st.cache_resource
def initialize_rag_application(vector_store_path, embedding_model_name, temperature, max_new_tokens, top_p):
    """
    Initializes and caches the RAGApplication object.
    """
    return RAGApplication(
        embedding_model_name=embedding_model_name,
        vector_store_path=vector_store_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p
    )

# Configure the page layout
st.set_page_config(layout="wide")

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

    # Sidebar: System Prompt
    st.sidebar.title("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Customize your prompt:",
        value="You are a helpful assistant. Answer the question as detailed as possible from the provided context. "
              "If the answer is not in the provided context, just say, 'answer is not available in the context'.",
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
            vector_database = st.selectbox("Database:", ["ChromaDB", "FAISS"], help="Default database set to ChromaDB")
            vector_store_path = st.text_input("Vector Store Path", "data/vectorstore/my_store", help="Path to vector database storage")

        # Model Selection
        with st.expander(label="Model Selection:", expanded=False):
            embedding_model_name = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "multilingual-e5-large"], help="Default set to all-MiniLM-L6-v2")
            vlm_model = st.selectbox("Vision Language Model:", ["Llama-3.2-11B-Vision-Instruct", "Phi-3.5-vision-instruct"], help="Default set to Llama-3.2-11B-Vision-Instruct")
            llm_model = st.selectbox("Large Language Model", ["Llama-3.2-3B-Instruct", "Llama-3.1-70B-Instruct"], help="Default set to Llama-3.2-3B-Instruct")

        # Document Settings
        with st.expander(label="Document Settings:", expanded=False):
            chunk_text = st.checkbox("Chunk Text", value=True)
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
            num_docs = st.number_input("Number of Documents to Retrieve", min_value=1, max_value=10, value=5, step=1)

        # LLM Settings
        with st.expander(label="LLM Settings:", expanded=False):
            max_new_tokens = st.number_input("Max New Tokens", min_value=50, max_value=65536, value=32768, step=50)
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
#            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            top_p = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
#            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    # Main Content Area
    with col2:
        
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
                    vector_store_path,
                    embedding_model_name,
                    temperature,
                    max_new_tokens,
                    top_p
                )

                # Load vector store if not already loaded
                if not hasattr(st.session_state, "vector_store"):
                    st.session_state.vector_store = rag.load_vector_store()

                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": num_docs})
                response = rag.get_conversational_chain(
                    retriever, user_question, llm_model, system_prompt
                )

                # Process response
                if response and "result" in response:
                    result = response["result"]
                    source_documents = response.get("source_documents", [])

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
                    st.write("### Source Documents")
                    for doc in source_documents:
                        metadata = doc.metadata
                        st.write(f"**Source:** {metadata.get('source', 'Unknown')}, "
                                 f"**Page Number:** {metadata.get('page_number', 'N/A')}")

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
                        st.write(f"**Source:** {metadata.get('source', 'Unknown')}, "
                                 f"**Page Number:** {metadata.get('page_number', 'N/A')}")

    # Process documents on submission
    if st.sidebar.button("Submit & Process"):
        try:
            # Initialize RAG application
            rag = initialize_rag_application(
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
                        rag.store_vector_data(text_chunks)
                        st.sidebar.success("Documents processed and stored successfully.")

        except Exception as e:
            st.sidebar.error(f"An error occurred while processing the files: {str(e)}")

if __name__ == "__main__":
    main()
