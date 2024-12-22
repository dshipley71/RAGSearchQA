import os
import json
import tempfile
import torch

from datetime import datetime
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import FAISS
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from pprint import pprint
from langchain.globals import set_debug

set_debug(False)

device = "cuda" if torch.cuda.is_available() else "cpu"

class RAGApplication:
    """
    A Retrieval-Augmented Generation (RAG) application for handling document processing,
    vector storage, and conversational AI workflows.
    """

    def __init__(self, embedding_model_name, vector_database, vector_store_path, temperature, max_new_tokens, top_p):
        """
        Initializes the RAG application with the specified parameters.

        Args:
            embedding_model_name (str): Name of the HuggingFace embedding model.
            vector_store_path (str): Path to store the vector database.
            temperature (float): Sampling temperature for the language model.
            max_new_tokens (int): Maximum number of new tokens generated by the language model.
            top_p (float): Top-p sampling for controlling diversity in generation.
        """
        self.embedding_model_name = embedding_model_name
        self.vector_database = vector_database
        self.vector_store_path = vector_store_path
        self.vector_store = None  # Cache for vector store optimization
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.cached_llm = None  # Cache for LLM resources

    def read_data(self, files):
        """
        Reads and processes data from uploaded files using UnstructuredLoader.

        Args:
            files (list): List of file-like objects to be processed.

        Returns:
            list: List of Document objects containing the processed data.

        Raises:
            ValueError: If there is an error processing any file.
        """
        documents = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name

            try:
                loader = UnstructuredLoader(
                    tmp_file_path,
                    chunking_strategy="basic", #"by_title", #"basic",
                    max_characters=16384,
                    overlap=256,
                    overlap_all=True,
                    include_orig_elements=False,
                    post_processors=[clean_extra_whitespace, group_broken_paragraphs],
                )
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
            except Exception as e:
                raise ValueError(f"Error processing file {file.name}: {str(e)}")
            finally:
                os.remove(tmp_file_path)
        return documents

    def get_chunks(self, texts, chunk_size, chunk_overlap):
        """
        Splits text data into chunks for processing.

        Args:
            texts (list): List of Document objects containing the text data.
            chunk_size (int): Maximum size of each chunk in characters.
            chunk_overlap (int): Number of overlapping characters between consecutive chunks.

        Returns:
            list: List of Document objects with text split into chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # chunks = [
        #     Document(page_content=split_text, metadata=text.metadata)
        #     for text in texts
        #     for split_text in text_splitter.split_text(text.page_content)
        # ]

        chunks = []

        for text in texts:
            split_texts = text_splitter.split_text(text.page_content)
            for split_text in split_texts:
                chunk = Document(page_content=split_text, metadata=text.metadata)
                chunks.append(chunk)

        return chunks

    def store_vector_data(self, text_chunks, embedding_model_name=None):
        """
        Stores text chunks in a vector store for later retrieval.

        Args:
            text_chunks (list): List of Document objects containing text chunks.
            embedding_model_name (str, optional): Name of the embedding model to use. Defaults to None.
            database (str): The vector database to use. Either 'FAISS' or 'ChromaDB'. Defaults to 'FAISS'.

        Returns:
            None
        """
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        if self.vector_database == "FAISS":
            vector_store = FAISS.from_documents(
                documents=filter_complex_metadata(text_chunks),
                embedding=embeddings
            )
            self.vector_store = vector_store  # Cache the vector store
        elif self.vector_database == "ChromaDB":
            vector_store = Chroma.from_documents(
                documents=filter_complex_metadata(text_chunks),
                embedding=embeddings,
                persist_directory=self.vector_store_path
            )
            self.vector_store = vector_store  # Cache the vector store
        else:
            raise ValueError(f"Unsupported database: {database}")

    def load_vector_store(self):
        """
        Loads the vector store from the specified path or in-memory.

        Args:
            database (str): The vector database to load. Either 'FAISS' or 'ChromaDB'. Defaults to 'FAISS'.

        Returns:
            VectorStore: The loaded vector store.
        """
        if self.vector_store is None:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            if self.vector_database == "FAISS":
                raise RuntimeError("FAISS is in-memory only and must be initialized with data.")
            elif self.vector_database == "ChromaDB":
                self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=embeddings
                )
            else:
                raise ValueError(f"Unsupported database: {database}")
        return self.vector_store

    def save_conversation(self, conversation):
        """
        Saves conversation history to a JSON file.

        Args:
            conversation (list): List of conversation messages to be saved.

        Returns:
            None
        """
        conversation_path = os.path.join(self.vector_store_path, "conversation_history.json")
        os.makedirs(self.vector_store_path, exist_ok=True)
        with open(conversation_path, "w") as f:
            json.dump(conversation, f, indent=4)

    def load_conversation(self):
        """
        Loads conversation history from a JSON file.

        Returns:
            list: List of conversation messages if available; otherwise, an empty list.
        """
        conversation_path = os.path.join(self.vector_store_path, "conversation_history.json")
        if os.path.exists(conversation_path):
            with open(conversation_path, "r") as f:
                return json.load(f)
        return []

    @staticmethod
    def document_to_dict(doc):
        """Converts a Document object to a serializable format."""
        return {"metadata": doc.metadata}
    def initialize_llm(self, llm_model):
        """
        Initializes and caches the LLM model, tokenizer, and pipeline.

        Args:
            llm_model (str): The name of the HuggingFace LLM model.

        Returns:
            HuggingFacePipeline: A cached pipeline for text generation.
        """
        if self.cached_llm is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype= "bfloat16"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model,
                local_files_only=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                # quantization_config=bnb_config
                local_files_only=True,
                low_cpu_mem_usage=True
            ).to(device)
            
            pipe = pipeline(
                "text-generation",
                model=model,
                device=device,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.15,
                return_full_text=False,
            )
            
            self.cached_llm = HuggingFacePipeline(pipeline=pipe)
        return self.cached_llm

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # def get_conversational_chain(self, retriever, ques, llm_model, system_prompt):
    def get_conversational_chain(self, ques, llm_model, system_prompt):
        """
        Generates a conversational chain response using an LLM and retriever.

        Args:
            retriever (BaseRetriever): The retriever to use for document retrieval.
            ques (str): The question or input query from the user.
            llm_model (str): The name of the HuggingFace LLM model.
            system_prompt (str): The system prompt to set the context for the conversation.

        Returns:
            dict: The response from the conversational chain, including source documents.
        """
        llm = self.initialize_llm(llm_model)

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
            }
        )

        prompt = PromptTemplate(
            template = system_prompt,
            input_variables=["context", "question", "history"],
        )
        # pprint(prompt)

        qa_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        qa_chain_with_source = RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
        ).assign(answer=qa_chain_from_docs)

        answer = qa_chain_with_source.invoke(ques)
        # pprint(answer)

        sources = answer["context"]

        # for index, source in enumerate(sources):
        #     print(f"{index}: source: {source.metadata['source']} | Page number: {source.metadata['page_number']} | File Type: {source.metadata['filetype']}")
        
        return answer