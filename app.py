import os
import streamlit as st
from datetime import datetime
from rag_application import RAGApplication

@st.cache_resource
def initialize_rag_application(vector_store_path, embedding_model_name, temperature, max_new_tokens, top_p):
    """
    Initializes and caches the RAGApplication object.

    Args:
        vector_store_path (str): Path to the vector store directory.
        embedding_model_name (str): Name of the embedding model.
        temperature (float): Sampling temperature for the language model.
        max_new_tokens (int): Maximum number of tokens to generate.
        top_p (float): Top-p sampling for controlling diversity in generation.

    Returns:
        RAGApplication: Cached instance of the RAGApplication.
    """
    return RAGApplication(
        embedding_model_name=embedding_model_name,
        vector_store_path=vector_store_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p
    )

def main():
    """
    Main function to run the Streamlit application for chatting with files using RAG and Llama-3.2.

    It provides a user interface to upload files, configure parameters, and ask questions related to the content
    of the files.
    """
    st.set_page_config(page_title="Chat with Your Files")
    st.header("Chat with Your Files using RAG and Llama-3.2")

    # Sidebar configurations
    vector_store_path = st.sidebar.text_input("Vector Store Path", "data/vectorstore/my_store")
    embedding_model_name = st.sidebar.selectbox("Select Embedding Model", ["all-MiniLM-L6-v2", "multilingual-e5-large"])
    llm_model = st.sidebar.selectbox("Select LLM Model", ["Llama-3.2-3B-Instruct"])
    chunk_text = st.sidebar.checkbox("Chunk Text", value=True)
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    num_docs = st.sidebar.number_input("Number of Documents to Retrieve", min_value=1, max_value=10, value=5, step=1)
    max_new_tokens = st.sidebar.number_input("Max New Tokens", min_value=50, max_value=65536, value=32768, step=50)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value="You are a helpful assistant. Answer the question as detailed as possible from the provided context. "
              "If the answer is not in the provided context, just say, 'answer is not available in the context'."
    )

    # Initialize and cache RAGApplication
    rag = initialize_rag_application(vector_store_path, embedding_model_name, temperature, max_new_tokens, top_p)

    # Document upload and processing
    st.sidebar.title("Upload Documents")
    data_files = st.sidebar.file_uploader("Upload your Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        if not data_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing..."):
                try:
                    raw_documents = rag.read_data(data_files)
                    if not raw_documents:
                        st.error("No valid documents were processed. Please check your files.")
                        return
                    text_chunks = (
                        rag.get_chunks(raw_documents, chunk_size, chunk_overlap) if chunk_text else raw_documents
                    )
                    rag.store_vector_data(text_chunks)
                    st.sidebar.success("Documents processed and stored successfully.")
                except Exception as e:
                    st.sidebar.error(f"An error occurred while processing the files: {str(e)}")

    # User interaction area
    user_question = st.text_input("Ask a Question from the Files")
    if user_question:
        try:
            if not hasattr(st.session_state, "vector_store"):
                st.session_state.vector_store = rag.load_vector_store()

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": num_docs})
            response = rag.get_conversational_chain(
                retriever, user_question, llm_model, system_prompt
            )

            # Load conversation history
            conversation = rag.load_conversation()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Handle response
            if response and "result" in response:
                result = response["result"]
                source_documents = response.get("source_documents", [])
                conversation.append({
                    "question": user_question,
                    "answer": result,
                    "timestamp": timestamp,
                    "llm_model": llm_model,
                    "source_documents": [rag.document_to_dict(doc) for doc in source_documents],
                })

                # Display response and source documents
                st.write("Reply: ", result)
                st.write(f"**LLM Model:** {llm_model}")
                st.write("### Source Documents")
                for doc in source_documents:
                    metadata = doc.metadata
                    st.write(f"**Source:** {metadata.get('source', 'Unknown')}, **Page Number:** {metadata.get('page_number', 'N/A')}, **Additional Info:** {metadata}")
            else:
                st.error("No result found. Please try again.")
                conversation.append({
                    "question": user_question,
                    "answer": "No result found.",
                    "timestamp": timestamp,
                    "llm_model": llm_model,
                })

            # Save updated conversation history
            rag.save_conversation(conversation)

            # Display conversation history
            st.write("### Conversation History")
            for entry in sorted(conversation, key=lambda x: x["timestamp"], reverse=True):
                with st.expander(f"Q: {entry['question']} ({entry['timestamp']})"):
                    st.write(f"**A:** {entry['answer']}")
                    st.write(f"**LLM Model:** {entry['llm_model']}")
                    if "source_documents" in entry:
                        for doc in entry["source_documents"]:
                            st.write(f"**Source:** {doc['metadata'].get('source', 'Unknown')}, "
                                     f"**Page Number:** {doc['metadata'].get('page_number', 'N/A')}, "
                                     f"**Additional Info:** {doc['metadata']}")

        except Exception as e:
            st.error(f"An error occurred while retrieving the response: {str(e)}")

if __name__ == "__main__":
    main()
