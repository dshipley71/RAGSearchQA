import os
import sys
import torch
import rich
import warnings

# image ocr
import base64
import io
from PIL import Image

from pathlib import Path
from pprint import pprint
from haystack import Pipeline
from haystack import Document

# device utilities
from haystack.utils import ComponentDevice

# chroma vector database
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

# document processing
from haystack.components.converters import PyPDFToDocument, TextFileToDocument, CSVToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter

# embedders
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder

# rag prompt and answer
from haystack.components.builders import PromptBuilder
from haystack.components.builders import AnswerBuilder
# from haystack.components.readers import ExtractiveReader

# local huggingface models
from haystack.components.generators import HuggingFaceLocalGenerator

from transformers import AutoConfig

###############################################################################

warnings.filterwarnings("ignore", category=FutureWarning)

device ="cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# utilities
###############################################################################
def check_directory(path):
    """
    Check if a directory exists.

    Args:
        path (str or pathlib.Path): The path to the directory.

    Returns:
        bool: True if the directory exists. False otherwise.
    """
    return Path(path).is_dir()

def encode_image_to_base64(image_path: str, format: str="PNG") -> str:
    """
    Encode an image file to a base64 string.;

    Args:
        image_path (str): Path to the image file.
        format (str): Formaty to save the image in memory (default is PNG)

    Returns:
        str: Base64-encoded image.

    Reference:
        Medium: Llama 3.2-Vision for High-Precision OCR with Ollama
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
###############################################################################
#
###############################################################################
data =[
    "test/steelers.txt",
]

audio_files = [
    "",
]
image_files = [
    "",
]

###############################################################################
# prompt template
###############################################################################
template = \
"""
You are a helpful assistant. Answer the question as detailed as possible from
the provided context and conversation history. If the answer is not in the
provided context, just say, 'answer is not available in the context'.

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

###############################################################################
# Create Document Store
###############################################################################
def create_document_store(
        collection_name="collection",
        persist_path="data/mycollection"
): 
    document_store = ChromaDocumentStore(
        collection_name=collection_name,
        persist_path=persist_path
    )

    return document_store

###############################################################################
# Document Processing
###############################################################################
def run_embedder(
    document_store,
    embedding_model="multilingual-e5-large",
    data_path="test/",
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=True,
    split_by="word",
    split_length=150,
    split_overlap=50,
    policy=DuplicatePolicy.SKIP
):
    # initialize converters
    pdf_converter = PyPDFToDocument()
    csv_converter = CSVToDocument()
    txt_converter = TextFileToDocument()

    # combine into a list of documents
    document_joiner = DocumentJoiner()

    # connect files to the proper converter
    file_type_router = FileTypeRouter(
        mime_types=["text/plain", "application/pdf", "text/csv"]
    )

    # create components
    document_cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True
    )

    document_splitter = DocumentSplitter(
        split_by="word",
        split_length=150,
        split_overlap=50
    )

    document_embedder = SentenceTransformersDocumentEmbedder(
        # model="models/all-MiniLM-L6-v2",
        model="models/multilingual-e5-large",
        device=ComponentDevice.from_str(device)
    )

    document_writer = DocumentWriter(
        document_store,
        policy=DuplicatePolicy.SKIP
    )

    document_embedder.warm_up()

    # add components to pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    indexing_pipeline.add_component(instance=txt_converter, name="txt_converter")
    indexing_pipeline.add_component(instance=pdf_converter, name="pdf_converter")
    indexing_pipeline.add_component(instance=csv_converter, name="csv_converter")
    indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    indexing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    indexing_pipeline.add_component(instance=document_writer, name="document_writer")
    
    # connect components
    indexing_pipeline.connect("file_type_router.text/plain", "txt_converter.sources")
    indexing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")
    indexing_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
    indexing_pipeline.connect("txt_converter", "document_joiner")
    indexing_pipeline.connect("csv_converter", "document_joiner")
    indexing_pipeline.connect("pdf_converter", "document_joiner")
    indexing_pipeline.connect("document_joiner", "document_cleaner")
    indexing_pipeline.connect("document_cleaner", "document_splitter")
    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    # run pipeline
    indexing_pipeline.run(
        {
            "file_type_router":
            {
                "sources": list(Path("test").glob("**/*"))
            }
        }
    )
    
    print("indexiong complete")

###############################################################################
# Chroma document store
###############################################################################
def run_rag(
        document_store,
):
    model_path = "models/Llama-3.2-3B"
    rich.print(model_path)

    # load and fix model configuration
    config = AutoConfig.from_pretrained(model_path)
    # config.rope_scaling = {
    #     "type": "linear",
    #     "factor": 32.0
    # }

    llm = HuggingFaceLocalGenerator(
        model=model_path,
        task="text-generation",
        huggingface_pipeline_kwargs={
            "device_map": device,
            "torch_dtype": torch.bfloat16
        },
        generation_kwargs={"max_new_tokens": 256}
    )

    llm.warm_up()

    # quick check
    rich.print(llm.run("What is the capital of Virgina?"))

    # create components
    embedder = SentenceTransformersTextEmbedder(
        # model="models/all-MiniLM-L6-v2",
        model="models/multilingual-e5-large",
        device=ComponentDevice.from_str(device)
    )

    retriever = ChromaEmbeddingRetriever(
        document_store=document_store
    )

    prompt_builder = PromptBuilder(
        template=template
    )

    answer_builder = AnswerBuilder()

    # add components to rag piupeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=embedder, name="embedder")
    rag_pipeline.add_component(instance=retriever, name="retriever")
    rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    rag_pipeline.add_component(instance=answer_builder, name="answer_builder")
    rag_pipeline.add_component(instance=llm, name="llm")

    # connect rag components
    rag_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    # rag_pipeline.connect("llm.meta", "answer_builder.replies") # TODO: add metadata when preprocessing documentation
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    question = (
        "What are the chances the Steelers win the AFC North?"
    )

    results = rag_pipeline.run(
        {
            "embedder": {
                "text": question
            },
            "prompt_builder": {
                "question": question
            },
            "answer_builder": {
                "query": question
            },
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 500,
                    "temperature": 0.1,
                    "top_k": 5,
                    "top_p": 0.95,
                    "repetition_penalty": 1.15,
                    "return_full_text": False,
                },
            }
        }
    )

    rich.print(results)

    print("RAG pipeline complete")

if __name__ == "__main__":
    doc_store = create_document_store()
    # run_embedder(doc_store)
    run_rag(doc_store)