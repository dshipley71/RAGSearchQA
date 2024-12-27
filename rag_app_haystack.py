###############################################################################
#
###############################################################################
import os
import torch
import rich
import warnings
import base64
import io

from PIL import Image
from pathlib import Path
from pprint import pprint

from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument, TextFileToDocument, CSVToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.builders import AnswerBuilder
from haystack.components.generators import HuggingFaceLocalGenerator

from transformers import AutoConfig, BitsAndBytesConfig

###############################################################################
#
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
# prompt template
###############################################################################
prompt_template = \
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
# Retrieval Augmented Generation (RAG) Pipeline
###############################################################################
class RAGApplication:

    # TODO: Use configparser to load configuration parameters
    # TODO: Replace converters with Unstructured IO
    # TODO: Input can be file or directory
    # TODO: Create prompt.config file for custom prompts or use DSPy
    # TODO: Utility functions into utils.py
    # TODO: Add error checking
    # TODO: Add logging
    # TODO: cache models to prevent reload

    def __init__(
        self,
        data_path="test/",
        template=prompt_template,
        collection_name="collection",
        persist_path=None, # None for non-persistent database, otherwise peristent
        embedding_model="models/multilingual-e5-large",
        llm_model="models/Llama-3.2-3B-Instruct",
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=True,
        split_by="word",
        split_length=150,
        split_overlap=50,
        policy=DuplicatePolicy.SKIP,
        task="text-generation",
        max_new_tokens=500,
        temperature=0.1,
        top_k=5,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False,
    ):
        self.data_path=data_path
        self.template = template
        self.collection_name=collection_name
        self.persist_path=persist_path
        self.embedding_model=embedding_model
        self.llm_model=llm_model
        self.remove_empty_lines=remove_empty_lines
        self.remove_extra_whitespaces=remove_extra_whitespaces
        self.remove_repeated_substrings=remove_repeated_substrings
        self.split_by=split_by
        self.split_length=split_length
        self.split_overlap=split_overlap
        self.policy=policy
        self.task=task
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p
        self.repetition_penalty=repetition_penalty
        self.return_full_text=return_full_text

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype= torch.bfloat16
        )

        if not check_directory(self.llm_model):
            raise FileNotFoundError(f"LLM model path not found: {self.llm_model}")

        if not check_directory(self.embedding_model):
            raise FileNotFoundError(f"Embedding model path not found: {self.embedding_model}")

        if not check_directory(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        if self.persist_path is not None:
            if not check_directory(self.persist_path):
                os.makedirs(self.persist_path, exist_ok=True)

    def create_document_store(self): 
        """
        Create document store.
        """
        self.document_store = ChromaDocumentStore(
            collection_name=self.collection_name,
            persist_path=self.persist_path
        )

    def run_embedder(self):
        """
        Document processing.
        """
        try:
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
                remove_empty_lines=self.remove_empty_lines,
                remove_extra_whitespaces=self.remove_extra_whitespaces,
                remove_repeated_substrings=self.remove_repeated_substrings
            )

            document_splitter = DocumentSplitter(
                split_by=self.split_by,
                split_length=self.split_length,
                split_overlap=self.split_overlap
            )

            document_embedder = SentenceTransformersDocumentEmbedder(
                model=self.embedding_model,
                device=ComponentDevice.from_str(device)
            )

            document_writer = DocumentWriter(
                self.document_store,
                policy=self.policy
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
                        "sources": list(Path(self.data_path).glob("**/*"))
                    }
                }
            )
            
            print("indexiong complete")

        except Exception as e:
            print(f"Error during embedding: {e}")

    def run_rag(self, question):
        """
        Retrieval Augmented Generation pipeline.
        """
        # config = AutoConfig.from_pretrained(self.llm_model)


        llm = HuggingFaceLocalGenerator(
            model=self.llm_model,
            task=self.task,
            huggingface_pipeline_kwargs={
                "device_map": device,
                "torch_dtype": torch.bfloat16,
                "model_kwargs": {
                    "quantization_config": self.bnb_config,
                },
            },
            generation_kwargs={
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "return_full_text": self.return_full_text,
            },
        )

        llm.warm_up()

        # quick check
        # rich.print(llm.run("What is the capital of Virgina?"))

        # create components
        embedder = SentenceTransformersTextEmbedder(
            model=self.embedding_model,
            device=ComponentDevice.from_str(device)
        )

        retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store
        )

        prompt_builder = PromptBuilder(
            template=self.template
        )

        answer_builder = AnswerBuilder()

        # add components to rag pipeline
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

        # run rag pipeline
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
                    # "generation_kwargs": {
                    #     "max_new_tokens": 500,
                    #     "temperature": 0.1,
                    #     "top_k": 5,
                    #     "top_p": 0.95,
                    #     "repetition_penalty": 1.15,
                    #     "return_full_text": False,
                    # },
                }
            }
        )

        rich.print(results)

def run_pipeline(question):
    """
    Run RAG pipeline
    """
    rag = RAGApplication()
    rag.create_document_store()
    rag.run_embedder()
    rag.run_rag(question)

if __name__ == "__main__":
    run_pipeline("Elaborate on the Multi-Armed Bandit problem?")