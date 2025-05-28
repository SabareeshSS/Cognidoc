"""
Cognidoc - Document Processing and Question Answering System
==========================================================

This module provides functionality for processing various types of documents (PDF, TXT, DOCX, images)
and answering questions about their content using local LLMs via Ollama.

Key Features:
- Multi-format document processing (PDF, TXT, DOCX, images)
- Text chunking and embedding generation
- Image description generation and embedding
- Question answering with context retrieval
- Document summarization
- Support for multiple LLM models

The system uses ChromaDB for vector storage and retrieval, and Langchain for document processing.

Sabareesh - 05/22/2025 - Version 1.0
"""

import os
import sys
import json
import uuid
import time
import logging
import tempfile
import chromadb
import fitz  # PyMuPDF
import ollama
from pathlib import Path # Added import statement

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Tuple, Union
from chromadb import Collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

SCRIPT_DIR = os.path.dirname(__file__) # Added SCRIPT_DIR definition

# Configure logging (moved earlier)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure paths are relative to this script's location or use absolute paths
# Use user's home directory for persistent storage to avoid permission issues
USER_HOME_DIR = Path.home()
CHROMA_PERSIST_DIR = USER_HOME_DIR / ".cognidoc_data" / "chroma_db_persistent"
COLLECTION_NAME = "user_doc_collection_electron_v1"

# Model Configuration - Initialize with None, will be set from models.json
TEXT_EMBEDDING_MODEL = None  # Will be set from models.json text_embedding_models
IMAGE_EMBEDDING_MODEL = None  # Will be set from models.json image_embedding_models
IMAGE_QUERY_MODEL = None     # Will be set from models.json querying_models
LLM_MODEL = None            # Will be set from models.json querying_models

# Processing Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TEXT_RETRIEVER_K = 50
IMAGE_RETRIEVER_K = 1

# Enhanced error handling for models.json
try:
    with open(os.path.join(SCRIPT_DIR, "models.json"), 'r') as f:
        models_data = json.load(f)
    categories = models_data.get('model_categories', {})

    # Validate model categories
    if not all(isinstance(categories.get(cat, []), list) for cat in ['querying_models', 'image_embedding_models', 'text_embedding_models']):
        raise ValueError("Invalid models.json structure: Expected lists for all model categories.")

    # Set default models from the first available model in each category
    TEXT_EMBEDDING_MODEL = categories.get('text_embedding_models', [])[0] if categories.get('text_embedding_models') else "all-minilm:latest"
    IMAGE_EMBEDDING_MODEL = categories.get('image_embedding_models', [])[0] if categories.get('image_embedding_models') else "llava:13b"
    IMAGE_QUERY_MODEL = categories.get('querying_models', [])[0] if categories.get('querying_models') else "llava-llama3:8b-v1.1-q4_0"
    LLM_MODEL = categories.get('querying_models', [])[0] if categories.get('querying_models') else "llava-llama3:8b-v1.1-q4_0"
except FileNotFoundError:
    logger.error("models.json not found. Using hardcoded defaults.")
    TEXT_EMBEDDING_MODEL = "all-minilm:latest"
    IMAGE_EMBEDDING_MODEL = "llava:13b"
    IMAGE_QUERY_MODEL = "llava-llama3:8b-v1.1-q4_0"
    LLM_MODEL = "llava-llama3:8b-v1.1-q4_0"
except (json.JSONDecodeError, ValueError) as e:
    logger.error(f"Error parsing models.json: {e}. Using hardcoded defaults.")
    TEXT_EMBEDDING_MODEL = "all-minilm:latest"
    IMAGE_EMBEDDING_MODEL = "llava:13b"
    IMAGE_QUERY_MODEL = "llava-llama3:8b-v1.1-q4_0"
    LLM_MODEL = "llava-llama3:8b-v1.1-q4_0"

# --- Global State ---
current_embedding_model: str = TEXT_EMBEDDING_MODEL
current_image_embedding_model: str = IMAGE_EMBEDDING_MODEL
current_querying_model: str = LLM_MODEL

# Dictionary to store state for each processed file
processed_files: Dict[str, Dict[str, Any]] = {}  # file_path -> state dict
# State will include:
# - processing_mode: str
# - file_name: str
# Ensure the Chroma directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --- Helper Functions ---
def is_image_file(file_path: Optional[str]) -> bool:
    """
    Check if a file is an image based on its extension.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file has an image extension, False otherwise
    """
    if not file_path:
        return False
    return os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

def _get_chroma_collection(persist_dir: str = CHROMA_PERSIST_DIR, 
                         collection_name: str = COLLECTION_NAME) -> Collection:
    """
    Get or create a ChromaDB collection for storing embeddings.
    
    Args:
        persist_dir: Directory where ChromaDB data is stored
        collection_name: Name of the collection to get or create
        
    Returns:
        Collection: ChromaDB collection object
    """
    client = chromadb.PersistentClient(path=str(persist_dir)) # Convert to string
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def _reset_collection_and_session(reason: str):
    """Clears the ChromaDB collection and the processed_files list."""
    global processed_files, COLLECTION_NAME, CHROMA_PERSIST_DIR
    logger.warning(f"Resetting ChromaDB collection ('{COLLECTION_NAME}') and processed files list due to: {reason}")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' deleted successfully.")
        # Recreate it immediately
        _get_chroma_collection()
        logger.info(f"Collection '{COLLECTION_NAME}' recreated successfully.")
    except Exception as e:
        logger.error(f"Error during collection reset for '{COLLECTION_NAME}': {e}. Manual check might be needed.", exc_info=True)
    processed_files.clear()
    logger.info("Processed files list has been cleared.")

def _generate_uuid(prefix: str = "") -> str:
    """
    Generate a unique identifier with an optional prefix.
    
    Args:
        prefix: Optional string prefix for the UUID
        
    Returns:
        str: Generated UUID with prefix if provided
    """
    return f"{prefix}_{uuid.uuid4()}"

def get_available_models() -> List[str]:
    """
    Get list of available models from models.json file.
    
    Returns:
        List[str]: List of available model names parsed from models.json categories
        
    Note:
        Models are now organized by categories in models.json under model_categories
    """
    try:
        models_file = os.path.join(SCRIPT_DIR, "models.json")
        if not os.path.exists(models_file):
            logger.warning("models.json not found, using default model list")
            default_models = ["llava-llama3:8b-v1.1-q4_0"]
            send_response({
                "type": "models_list",
                "models": default_models,
                "message": "Using default model list (models.json not found)"
            })
            return default_models

        with open(models_file, 'r') as f:
            models_data = json.load(f)
            
        # Extract all models from the model_categories
        available_models = []
        categories = models_data.get('model_categories', {})
        
        # Collect models from each category
        for category_name, category_models in categories.items():
            if isinstance(category_models, list):
                available_models.extend(category_models)
            
        if not available_models:
            logger.warning("No models found in models.json categories, using default model list")
            default_models = ["llava-llama3:8b-v1.1-q4_0"]
            send_response({
                "type": "models_list",
                "models": default_models,
                "message": "Using default model list (no models in models.json)"
            })
            return default_models

        # Remove duplicates while preserving order
        available_models = list(dict.fromkeys(available_models))
        
        logger.info(f"Loaded {len(available_models)} models from {len(categories)} categories in models.json")
        send_response({
            "type": "models_list",
            "models": available_models,
            "message": "Using models from models.json"
        })
        return available_models

    except Exception as e:
        logger.error(f"Error loading models from models.json: {e}", exc_info=True)
        error_message = f"Error loading models: {str(e)}. Check models.json format and file permissions. Using default model."
        send_response({
            "type": "error",
            "models": [],
            "message": error_message,
            "request_id": "unknown" # Add request_id for better tracking
        })
        return []

def set_llm_model(model_name: str) -> str:
    """Set the current LLM model (simplified version)."""
    global current_llm_model
    try:
        available_models = get_models_by_category("querying_models")
        
        if model_name in available_models:
            current_llm_model = model_name
            logger.info(f"Set LLM model to: {model_name}")
            return f"Successfully set model to {model_name}"
        else:
            logger.warning(f"Requested model {model_name} not in available models. Using {current_llm_model}")
            return f"Model {model_name} not available. Using {current_llm_model}"
    except Exception as e:
        logger.error(f"Error setting LLM model: {e}", exc_info=True)
        return f"Error setting model: {str(e)}. Using {current_llm_model}"

def set_embedding_model(model_name: str) -> str:
    """
    Set the current embedding model for document processing.
    
    Args:
        model_name: Name of the model to use for embeddings
        
    Returns:
        str: Success/error message indicating the result
        
    Note:
        The model must be available in the models.json list
    """
    global current_embedding_model
    try:
        if model_name == current_embedding_model:
            logger.info(f"Text embedding model is already set to: {model_name}")
            return f"Text embedding model already set to {model_name}."

        available_models = get_models_by_category("text_embedding_models")
        if model_name not in available_models:
            logger.warning(f"Requested text embedding model '{model_name}' not in available models. Using '{current_embedding_model}'.")
            return f"Model '{model_name}' not available for text embedding. Using '{current_embedding_model}'."

        _reset_collection_and_session(f"text embedding model changed from '{current_embedding_model}' to '{model_name}'")
        current_embedding_model = model_name
        logger.info(f"Set text embedding model to: {model_name}. Database and session reset.")
        return f"Successfully set text embedding model to {model_name}. Database and session have been reset."

    except Exception as e:
        logger.error(f"Error setting embedding model: {e}", exc_info=True)
        return f"Error setting model: {str(e)}. Using {current_embedding_model}"

def set_querying_model(model_name: str) -> str:
    """
    Set the current querying model for answering questions.
    
    Args:
        model_name: Name of the model to use for querying
        
    Returns:
        str: Success/error message indicating the result
        
    Note:
        The model must be available in the models.json list
    """
    global current_querying_model
    try:
        available_models = get_models_by_category("querying_models")
        
        if model_name in available_models:
            current_querying_model = model_name
            logger.info(f"Set querying model to: {model_name}")
            return f"Successfully set querying model to {model_name}"
        else:
            logger.warning(f"Requested model {model_name} not in available models. Using {current_querying_model}")
            return f"Model {model_name} not available. Using {current_querying_model}"
    except Exception as e:
        logger.error(f"Error setting querying model: {e}", exc_info=True)
        return f"Error setting model: {str(e)}. Using {current_querying_model}"

def set_image_embedding_model(model_name: str) -> str:
    """
    Set the current image embedding model for processing images.
    
    Args:
        model_name: Name of the model to use for image embeddings
        
    Returns:
        str: Success/error message indicating the result
        
    Note:
        The model must be available in the models.json list
    """
    global current_image_embedding_model
    try:
        if model_name == current_image_embedding_model:
            logger.info(f"Image embedding model is already set to: {model_name}")
            return f"Image embedding model already set to {model_name}."

        available_models = get_models_by_category("image_embedding_models")
        if model_name not in available_models:
            logger.warning(f"Requested image embedding model '{model_name}' not in available models. Using '{current_image_embedding_model}'.")
            return f"Model '{model_name}' not available for image embedding. Using '{current_image_embedding_model}'."

        _reset_collection_and_session(f"image embedding model changed from '{current_image_embedding_model}' to '{model_name}'")
        current_image_embedding_model = model_name
        logger.info(f"Set image embedding model to: {model_name}. Database and session reset.")
        return f"Successfully set image embedding model to {model_name}. Database and session have been reset."
    except Exception as e:
        logger.error(f"Error setting image embedding model: {e}", exc_info=True)
        return f"Error setting model: {str(e)}. Using {current_image_embedding_model}"

def get_models_by_category(category: str) -> List[str]:
    """
    Get list of models for a specific category from models.json.
    
    Args:
        category: Category name to get models for ('querying_models', 'image_embedding_models', 'text_embedding_models')
        
    Returns:
        List[str]: List of model names for the specified category
    """
    try:
        models_file = os.path.join(SCRIPT_DIR, "models.json")
        if not os.path.exists(models_file):
            logger.warning(f"models.json not found, using default model for {category}")
            if category == "querying_models":
                return ["llava-llama3:8b-v1.1-q4_0"]
            elif category == "image_embedding_models":
                return ["llava:13b"]
            else:  # text_embedding_models
                return ["all-minilm:latest"]

        with open(models_file, 'r') as f:
            models_data = json.load(f)
            
        categories = models_data.get('model_categories', {})
        models = categories.get(category, [])
        
        if not models:
            logger.warning(f"No models found for category {category}, using default")
            if category == "querying_models":
                return ["llava-llama3:8b-v1.1-q4_0"]
            elif category == "image_embedding_models":
                return ["llava:13b"]
            else:  # text_embedding_models
                return ["all-minilm:latest"]

        return models

    except Exception as e:
        logger.error(f"Error loading models for category {category}: {e}", exc_info=True)
        # Return appropriate default model based on category
        if category == "querying_models":
            return ["llava-llama3:8b-v1.1-q4_0"]
        elif category == "image_embedding_models":
            return ["llava:13b"]
        else:  # text_embedding_models
            return ["all-minilm:latest"]
# --- Core RAG Functions (mostly unchanged, minor logging adjustments) ---
# load_and_split_text_document, generate_text_embeddings_parallel,
# get_image_description, query_llava_with_image, query_text_llm,
# summarize_transcript
# (Keep these functions as they are in the original script, ensuring they use the logger)
# --- Placeholder for brevity - Assume these functions exist here ---
# --- (Copy the implementations from the original script) ---
# Example:
def _load_text_content_from_file(file_path: str) -> Optional[List[Document]]:
    """Loads a TEXT-BASED document (PDF, TXT, DOCX) and returns a list of Document objects."""
    logger.info(f"Loading text document: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    documents: Optional[List[Document]] = None

    try:
        if file_extension == ".pdf":
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()
        elif file_extension == ".docx":
            try:
                from docx import Document as DocxDocument # Local import
                doc = DocxDocument(file_path)
                text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
                if text_content.strip():
                    documents = [Document(page_content=text_content, metadata={"source": os.path.basename(file_path)})]
                else:
                    documents = []
            except ImportError:
                logger.error("python-docx not installed. Run 'pip install python-docx'. Cannot process DOCX.")
                return None
            except Exception as e:
                logger.error(f"Error processing DOCX file {file_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Unsupported text document type: {file_extension}")
            return None

        if documents is None:
             logger.error(f"Loader failed unexpectedly for {file_path}")
             return None
        if not documents:
            logger.warning(f"No content loaded from {file_path}. File might be empty or unreadable.")
            return []
        
        # Ensure source metadata is present
        for doc_obj in documents:
            if 'source' not in doc_obj.metadata:
                doc_obj.metadata['source'] = os.path.basename(file_path)
            # Ensure page numbers are correctly formatted if present (example for PyMuPDFLoader)
            if 'page' in doc_obj.metadata and isinstance(doc_obj.metadata['page'], int):
                 doc_obj.metadata['page_number'] = doc_obj.metadata['page'] + 1 # 0-indexed to 1-indexed

        logger.info(f"Successfully loaded {len(documents)} sections from {file_path}.")
        return documents

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
        return None

def load_and_split_text_document(documents: List[Document], chunk_size: int, chunk_overlap: int) -> Optional[List[Document]]:
    """
    Splits loaded documents into text chunks using RecursiveCharacterTextSplitter
    and cleans metadata using filter_complex_metadata.
    This function now expects a list of already loaded Document objects.
    """
    if not documents:
        logger.warning("No documents provided to load_and_split_text_document for splitting. Returning empty list.")
        return []

    logger.info(
        f"Starting to split {len(documents)} loaded document sections with chunk_size={chunk_size} "
        f"and chunk_overlap={chunk_overlap}."
    )
    try:

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        cleaned_chunks = filter_complex_metadata(chunks)

        if not cleaned_chunks:
            logger.warning(f"Splitting {len(documents)} documents resulted in zero valid chunks after cleaning.")
            return []
        logger.info(f"Split into {len(cleaned_chunks)} chunks.")
        return cleaned_chunks

    except Exception as e:
        logger.error(f"Error during document splitting/cleaning: {e}", exc_info=True)
        return None

def generate_text_embeddings_parallel(chunks: List[Document], embed_func: OllamaEmbeddings) -> List[Tuple[int, List[float]]]:
    """Generates embeddings for text document chunks in parallel. Returns empty list on failure."""
    if not chunks or not embed_func:
        logger.warning("No text chunks or embedding function provided for embedding.")
        return []

    logger.info(f"Starting text embedding for {len(chunks)} chunks using '{embed_func.model}'...")
    results_with_indices: List[Tuple[int, Optional[List[float]]]] = []

    def _embed_chunk_with_index(index_chunk_tuple: Tuple[int, Document]) -> Tuple[int, Optional[List[float]]]:
        index, chunk = index_chunk_tuple
        try:
            content = getattr(chunk, 'page_content', '')
            if not content or content.isspace():
                logger.warning(f"Skipping embedding for chunk {index} due to empty content.")
                return index, None
            embedding = embed_func.embed_query(content)
            return index, embedding
        except Exception as e:
            logger.error(f"Error embedding chunk {index} with model '{embed_func.model}': {type(e).__name__}", exc_info=False)
            return index, None

    try:
        with ThreadPoolExecutor(thread_name_prefix="EmbedWorker") as executor:
            results_with_indices = list(executor.map(_embed_chunk_with_index, enumerate(chunks)))

        successful_embeddings = [(idx, emb) for idx, emb in results_with_indices if emb is not None]
        failed_count = len(chunks) - len(successful_embeddings)

        if failed_count > 0:
            logger.warning(f"{failed_count} text chunks failed during embedding.")
        if not successful_embeddings and chunks:
             logger.error(f"Text embedding failed for all {len(chunks)} input chunks.")
             return []

        logger.info(f"Successfully embedded {len(successful_embeddings)} text chunks.")
        successful_embeddings.sort(key=lambda x: x[0])
        return successful_embeddings
    except Exception as e:
        logger.error(f"Error during parallel text embedding execution: {e}", exc_info=True)
        return []

def get_image_description(image_path: str, model: str = IMAGE_QUERY_MODEL) -> Optional[str]:
    """Uses LLaVA via ollama.chat to generate a description of an image."""
    if not os.path.exists(image_path):
        logger.error(f"Image file not found for description generation: {image_path}")
        return None
    try:
        logger.info(f"Generating description for image: {os.path.basename(image_path)} using {model}")
        prompt = "Describe this image in detail, focusing on objects, scene, text, and overall context."
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}]
        )
        description = response.get('message', {}).get('content')
        if description:
            logger.info(f"Generated description for {os.path.basename(image_path)} (length: {len(description)})")
            return description.strip()
        else:
            logger.error(f"Failed to get description from LLaVA for {os.path.basename(image_path)}. Response: {response}")
            return None
    except ollama.ResponseError as ore:
        logger.error(f"Ollama API Error generating description for '{os.path.basename(image_path)}' with '{model}': {ore.status_code} - {ore.error}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"Error generating image description for {os.path.basename(image_path)}: {e}", exc_info=True)
        return None

def query_llava_with_image(question: str, image_path: str, llm_model: str = None) -> str:
    """Queries the specified Multimodal model (LLaVA) with a question and an image."""
    # Use IMAGE_QUERY_MODEL if no model provided. Don't use current_llm_model for image queries as it may not be multimodal.
    model_to_use = llm_model or IMAGE_QUERY_MODEL
    logger.info(f"Querying Multimodal model '{model_to_use}' with image '{os.path.basename(image_path)}' and question: '{question[:50]}...'")

    if not os.path.exists(image_path):
        logger.error(f"Image file not found at path for querying: {image_path}")
        return "Error: The image file specified could not be found for answering the question."

    try:
        start_time = time.time()
        response = ollama.chat(
            model=model_to_use,
            messages=[{'role': 'user', 'content': question, 'images': [image_path]}]
        )
        message_content = response.get('message', {}).get('content')
        if message_content:
            end_time = time.time()
            logger.info(f"Multimodal model '{model_to_use}' answered in {end_time - start_time:.2f} seconds.")
            return message_content.strip()
        else:
            logger.error(f"Multimodal model response missing content. Full response: {response}")
            return "Error: Received incomplete response from the multimodal model."
    except ollama.ResponseError as ore:
        logger.error(f"Ollama API Error (Multimodal Query): {ore.status_code} - {ore.error}", exc_info=False)
        return f"Error: Multimodal model API error ({ore.status_code}). Check Ollama server/model '{model_to_use}'. Details: {ore.error}"
    except Exception as e:
        logger.error(f"Error querying multimodal model '{model_to_use}': {e}", exc_info=True)
        return f"Error querying the multimodal model ({type(e).__name__})."

def query_text_llm(question: str, context: str, llm_model: str = None) -> str:
    """Formats prompt with TEXT context and queries the specified Ollama LLM."""
    # Use current_llm_model if no specific model is provided
    model_to_use = llm_model or current_llm_model
    logger.info(f"Querying Text LLM '{model_to_use}' for question: '{question[:50]}...'")

    if not context or context.startswith("Error:") or context == "No relevant context found.":
        context_issue = context if context else "Context is missing"
        logger.warning(f"Context indicates an issue or is missing: '{context_issue}'. Cannot answer based on document.")
        return f"I couldn't find relevant information in the document to answer your question. ({context_issue})"

    template = """You are an AI assistant. Answer the following question based *only* on the provided context.
If the context does not contain the answer, state "The provided context does not contain the answer to this question."
Do not use any external knowledge. Keep the answer concise.

Context:
---
{context}
---

Question: {question}

Answer:"""
    formatted_prompt = template.format(context=context, question=question)

    try:
        start_time = time.time()
        response = ollama.chat(model=model_to_use, messages=[{'role': 'user', 'content': formatted_prompt}])
        message_content = response.get('message', {}).get('content')
        if message_content:
            end_time = time.time()
            logger.info(f"LLM '{model_to_use}' answered in {end_time - start_time:.2f} seconds.")
            return message_content.strip()
        else:
            logger.error(f"Text LLM response missing content. Full response: {response}")            
            return "Error: Received incomplete response from the language model."
    except ollama.ResponseError as ore:
        logger.error(f"Ollama API Error (Text LLM): {ore.status_code} - {ore.error}", exc_info=False)
        return f"Error: Text LLM API error ({ore.status_code}). Check Ollama server/model '{model_to_use}'. Details: {ore.error}"
    except Exception as e:
        logger.error(f"Error querying Text LLM model '{model_to_use}': {e}", exc_info=True)
        return f"Error querying the text language model ({type(e).__name__})."

def summarize_transcript(transcript: str, max_chunk_chars: int = 2000, summary_model: str = LLM_MODEL) -> str:
    """Summarizes text using the LLM, handling long text by chunking."""
    if not transcript or transcript.isspace():
        return "No text provided to summarize."

    chunks = [transcript[i:i + max_chunk_chars] for i in range(0, len(transcript), max_chunk_chars)]
    if not chunks:
        logger.error("Failed to create chunks from non-empty transcript.")
        return "Error: Could not process the text into chunks for summarization."

    logger.info(f"Split transcript into {len(chunks)} chunks for summarization.")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize the following text into concise key points (this is chunk {i+1} of {len(chunks)}):\n\nText Chunk:\n{chunk}\n\nSummary:"
        logger.info(f"Requesting summary for chunk {i+1}/{len(chunks)} using model {summary_model}")
        try:
            response = ollama.chat(model=summary_model, messages=[{'role': 'user', 'content': prompt}])
            content = response.get('message', {}).get('content')
            if content:
                 chunk_summaries.append(content.strip())
            else:
                 logger.warning(f"Received empty summary response for chunk {i+1}.")
                 chunk_summaries.append(f"[Summary for chunk {i+1} failed]")
        except Exception as e:
            logger.error(f"Error summarizing chunk {i+1} with model {summary_model}: {e}", exc_info=True)
            chunk_summaries.append(f"[Error summarizing chunk {i+1}]")

    combined_summary = "\n\n".join(chunk_summaries)

    if len(chunks) > 1 and len(combined_summary) > max_chunk_chars * 1.2:
        logger.info("Generating final meta-summary.")
        final_prompt = f"Synthesize the following chunk summaries into a single, coherent summary:\n\n{combined_summary}\n\nFinal Summary:"
        try:
             response = ollama.chat(model=summary_model, messages=[{'role': 'user', 'content': final_prompt}])
             content = response.get('message', {}).get('content')
             if content:
                  return content.strip()
             else:
                  logger.warning("Failed to generate final meta-summary. Returning combined chunk summaries.")
                  return combined_summary
        except Exception as e:
            logger.error(f"Error generating final meta-summary: {e}", exc_info=True)
            return combined_summary
    else:
        return combined_summary
# --- End Core Functions ---


# --- Document Processing (Modified for API context) ---
def _process_document_internal(file_path: str) -> Tuple[Optional[str], str]:
    """
    Internal logic for document processing.
    Returns: (processing_mode | None, status_message)
    """
    global processed_files, current_embedding_model, current_querying_model

    file_name = os.path.basename(file_path)
    logger.info(f"--- Starting Processing for: {file_name} ---")
    start_time = time.time()

    processing_mode: Optional[str] = None
    status_message = f"Processing '{file_name}'...\n"
    processed_item_count = 0

    # Use a temporary directory for processing that cleans up
    processing_temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = processing_temp_dir.name
    logger.info(f"Using temp directory for processing: {temp_dir_path}")

    try:
        # Check if this specific file_path has already been processed in the current session
        if file_path in processed_files:
            logger.info(f"File '{file_name}' (path: {file_path}) already processed in this session. Skipping re-processing.")
            existing_state = processed_files[file_path]
            return existing_state['processing_mode'], f"File '{file_name}' already processed in this session."

        collection: Collection = _get_chroma_collection()

        # Check if the document's content (based on file_name as source) is already in ChromaDB
        # from a previous session.
        existing_docs_in_db: Dict[str, Any] = collection.get(where={"source": file_name}, limit=1)
        if existing_docs_in_db and existing_docs_in_db.get("ids"):
            logger.info(f"Content from '{file_name}' found in ChromaDB (possibly from a previous session). Will not re-embed.")
            determined_mode = None
            if is_image_file(file_path):
                determined_mode = 'image'
            elif file_path.lower().endswith(".pdf"):
                determined_mode = 'pdf'
            elif file_path.lower().endswith((".txt", ".docx")):
                determined_mode = 'text'

            if determined_mode:
                status_msg = f"Content from '{file_name}' is already in the database and will be available for queries."
                # This file will be added to `processed_files` by the main_loop.
                return determined_mode, status_msg
            else:
                # Should not happen if file_path is valid and supported
                logger.warning(f"Could not determine mode for existing file {file_name} based on its extension, proceeding with full process attempt.")
                # Fall through to normal processing. If IDs are truly unique, it might add duplicates.
                # Or, if source is the key, it might update. Chroma's `add` with same IDs updates.
                # Here, we generate new UUIDs, so it would add new entries if we proceed.
                # For now, this path means we re-process if mode determination fails.

        # Initialize text embedder with current model
        text_embed_func: OllamaEmbeddings
        try:
            text_embed_func = OllamaEmbeddings(model=current_embedding_model)
            text_embed_func.embed_query("test") # Test connection
            logger.info(f"Text embedding model '{current_embedding_model}' initialized.")
        except Exception as e:
             logger.critical(f"Fatal: Failed to initialize text embedding model {current_embedding_model}: {e}", exc_info=True)
             status_message += f"Error: Failed to initialize text embedder '{current_embedding_model}'. Check Ollama server."
             raise RuntimeError(status_message) # Stop processing

        # --- IMAGE FILE ---
        if is_image_file(file_path):
            # Initialize image embedder with current model
            image_embed_func = OllamaEmbeddings(model=current_image_embedding_model)
            status_message += f"Detected Image file. Generating description...\n"
            description = get_image_description(file_path, model=current_image_embedding_model)

            if description:
                status_message += f"Generated description (len: {len(description)}). Embedding...\n"
                desc_embedding = image_embed_func.embed_query(description)
                if desc_embedding:
                    image_id = _generate_uuid("image_desc")
                    metadata = {
                        'source': file_name, 'type': 'image_description',
                        'original_path': file_path, # Store original path
                        'description': description[:500],
                        'embedding_model': current_image_embedding_model
                    }
                    collection.add(ids=[image_id], embeddings=[desc_embedding], metadatas=[metadata])
                    processed_item_count += 1
                    status_message += "Stored image description embedding.\n"
                else: status_message += f"Error: Failed to embed description.\n"; processing_mode = None
            else: status_message += f"Error: Failed to generate description.\n"; processing_mode = None

        # --- PDF FILE ---
        elif file_path.lower().endswith(".pdf"):
            processing_mode = 'pdf'
            status_message += f"Detected PDF file. Extracting images and text...\n"
            pdf_images_processed_count = 0
            pdf_text_chunks_processed_count = 0

            # --- PDF Image Handling ---
            try:
                pdf_document = fitz.open(file_path)
                status_message += f"Opened PDF ({len(pdf_document)} pages). Processing images...\n"
                logger.warning("Storing temporary paths for extracted PDF images. Querying images might fail if the backend restarts.")

                for page_num in range(len(pdf_document)):
                    image_list = pdf_document.get_page_images(page_num, full=True)
                    if not image_list: continue
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        if not base_image or not base_image.get("image"): continue

                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        img_filename = f"pdf_{os.path.splitext(file_name)[0]}_pg{page_num + 1}_img{img_index}_{uuid.uuid4()}.{image_ext}"
                        # Save to the processing temp dir
                        temp_image_path = os.path.join(temp_dir_path, img_filename)

                        try:
                            with open(temp_image_path, "wb") as img_file: img_file.write(image_bytes)
                            # NOTE: temp_image_path is stored in metadata. This path is only valid
                            # as long as the processing_temp_dir exists (i.e., during this request scope).
                            # For persistent image querying, copy to PDF_IMAGE_PERSIST_DIR and store that path.
                            image_path_to_store = temp_image_path

                            description = get_image_description(temp_image_path, model=current_image_embedding_model)
                            if description:
                                desc_embedding = image_embed_func.embed_query(description)
                                if desc_embedding:
                                    img_id = _generate_uuid("pdf_img_desc")
                                    metadata = {
                                        'source': file_name, 'type': 'image_description',
                                        'original_path': image_path_to_store,
                                        'description': description[:500],
                                        'embedding_model': current_image_embedding_model,
                                        'page_number': page_num + 1
                                    }
                                    collection.add(ids=[img_id], embeddings=[desc_embedding], metadatas=[metadata])
                                    pdf_images_processed_count += 1
                                else: status_message += f"Warn: Failed embed desc page {page_num+1}, img {img_index}.\n"
                            else: status_message += f"Warn: Failed get desc page {page_num+1}, img {img_index}.\n"
                        except Exception as img_proc_err:
                            logger.error(f"Failed to save/process extracted image pg {page_num + 1}, idx {img_index}: {img_proc_err}")

                processed_item_count += pdf_images_processed_count
                status_message += f"Processed {pdf_images_processed_count} PDF images (descriptions embedded).\n"
            except Exception as pdf_img_err:
                logger.error(f"Error during PDF image extraction/processing: {pdf_img_err}", exc_info=True)
                status_message += f"Warning: Error extracting PDF images: {type(pdf_img_err).__name__}.\n"

            # --- PDF Text Handling ---
            status_message += "Processing text...\n"
            loaded_documents = _load_text_content_from_file(file_path)
            if loaded_documents:
                status_message += f"Loaded {len(loaded_documents)} sections from PDF. Splitting...\n"
                text_chunks = load_and_split_text_document(loaded_documents, CHUNK_SIZE, CHUNK_OVERLAP)
                if not text_chunks:
                    status_message += "Warning: Failed to split PDF text content or no content after splitting.\n"
                else:
                    status_message += f"Split PDF into {len(text_chunks)} text chunks. Embedding...\n"
                embedding_results = generate_text_embeddings_parallel(text_chunks, text_embed_func)
                if embedding_results:
                    ids_to_add, embeddings_to_add, metadatas_to_add, documents_to_add = [], [], [], []
                    valid_indices = {idx for idx, emb in embedding_results}
                    for i, chunk in enumerate(text_chunks):
                        if i in valid_indices:
                            embedding = next(emb for idx, emb in embedding_results if idx == i)
                            chunk_id = _generate_uuid("pdf_text")
                            metadata = chunk.metadata or {}
                            serializable_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in metadata.items()}
                            serializable_meta['type'] = 'text'
                            serializable_meta['embedding_model'] = current_embedding_model
                            if 'page' in serializable_meta:
                                try: # Ensure page is int
                                     serializable_meta['page_number'] = int(serializable_meta['page']) + 1
                                except (ValueError, TypeError):
                                     logger.warning(f"Could not parse page number '{serializable_meta.get('page')}' for chunk {i}")
                                     serializable_meta['page_number'] = serializable_meta.get('page', 'unknown') # Keep original if parsing fails

                            ids_to_add.append(chunk_id)
                            embeddings_to_add.append(embedding)
                            metadatas_to_add.append(serializable_meta)
                            documents_to_add.append(chunk.page_content)

                    if ids_to_add:
                        collection.add(ids=ids_to_add, embeddings=embeddings_to_add, metadatas=metadatas_to_add, documents=documents_to_add)
                        pdf_text_chunks_processed_count = len(ids_to_add)
                        processed_item_count += pdf_text_chunks_processed_count
                        status_message += f"Stored {pdf_text_chunks_processed_count} text chunk embeddings.\n"
                        logger.info(f"Stored {pdf_text_chunks_processed_count} PDF text embeddings.")
                    else: status_message += "Warning: No valid text embeddings generated for PDF.\n"
                else: status_message += "Warning: Failed to generate embeddings for PDF text chunks.\n"
            else: status_message += "No text content found/extracted from PDF.\n"

        # --- OTHER TEXT FILES (TXT, DOCX) ---
        elif file_path.lower().endswith((".txt", ".docx")):
            processing_mode = 'text'
            status_message += f"Detected {processing_mode.upper()} file. Loading, splitting, embedding...\n"
            loaded_documents = _load_text_content_from_file(file_path)
            if loaded_documents:
                status_message += f"Loaded {len(loaded_documents)} sections from {processing_mode.upper()} file. Splitting...\n"
                text_chunks = load_and_split_text_document(loaded_documents, CHUNK_SIZE, CHUNK_OVERLAP)
                if not text_chunks:
                    status_message += f"Warning: Failed to split {processing_mode.upper()} content or no content after splitting.\n"
                else:
                    status_message += f"Split {processing_mode.upper()} file into {len(text_chunks)} text chunks. Embedding...\n"
                embedding_results = generate_text_embeddings_parallel(text_chunks, text_embed_func)
                if embedding_results:
                    ids_to_add, embeddings_to_add, metadatas_to_add, documents_to_add = [], [], [], []
                    valid_indices = {idx for idx, emb in embedding_results}
                    for i, chunk in enumerate(text_chunks):
                        if i in valid_indices:
                            embedding = next(emb for idx, emb in embedding_results if idx == i)
                            chunk_id = _generate_uuid(f"{processing_mode}_text")
                            metadata = chunk.metadata or {}
                            serializable_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in metadata.items()}
                            serializable_meta['type'] = 'text'
                            serializable_meta['embedding_model'] = current_embedding_model
                            ids_to_add.append(chunk_id)
                            embeddings_to_add.append(embedding)
                            metadatas_to_add.append(serializable_meta)
                            documents_to_add.append(chunk.page_content)

                    if ids_to_add:
                        collection.add(ids=ids_to_add, embeddings=embeddings_to_add, metadatas=metadatas_to_add, documents=documents_to_add)
                        text_items_processed = len(ids_to_add)
                        processed_item_count += text_items_processed
                        status_message += f"Stored {text_items_processed} text chunk embeddings.\n"
                        logger.info(f"Stored {text_items_processed} {processing_mode.upper()} text embeddings.")
                    else: status_message += "Warning: No valid text embeddings generated.\n"
                else: status_message += "Error: Failed to generate embeddings.\n"; processing_mode = None
            else: status_message += f"No text content loaded from {processing_mode.upper()} file.\n"

        else:
            status_message += f"Error: Unsupported file type '{os.path.splitext(file_name)[1]}'.\n"
            processing_mode = None

        # --- Finalize ---
        if processing_mode is not None:
            end_time = time.time()
            processing_time = end_time - start_time
            status_message += f"Processing complete ({processed_item_count} items stored). Took {processing_time:.2f} seconds.\n"
            status_message += f"Ready to answer questions about '{file_name}'!"
            logger.info(f"--- Processing Complete for: {file_name} (Mode: {processing_mode}, Items: {processed_item_count}) ---")
        else:
            if not status_message.endswith("failed.\n"): status_message += "Processing failed.\n"
            logger.error(f"--- Processing Failed for: {file_name} ---")

    except Exception as e:
        logger.critical(f"Unhandled error during document processing for {file_name}: {e}", exc_info=True)
        status_message += f"Critical Error: {type(e).__name__}. Check logs.\nProcessing failed."
        processing_mode = None
    finally:
        # IMPORTANT: Clean up the temporary directory for this processing run
        processing_temp_dir.cleanup()
        logger.info(f"Cleaned up temp directory: {temp_dir_path}")

    return processing_mode, status_message

# --- Query Processing (Modified for API context) ---
def _process_response_internal(query: str) -> str:
    """Internal logic for handling user queries across multiple processed documents."""
    global processed_files, current_embedding_model, current_querying_model

    if not query or query.isspace():
        return "Please enter a query or ask for a summary."
    if not processed_files:
        return "Error: No documents have been processed yet. Please upload and process files."

    logger.info(f"Processing query: '{query[:50]}...' for {len(processed_files)} files")

    # 1. Summarization Request
    is_summarization_request = any(keyword in query.lower() for keyword in ["summarize", "summary", "minutes", "key points"])

    if is_summarization_request:
        logger.info("Summarization request detected.")
        summaries = []
        for file_path, state in processed_files.items():
            mode = state['processing_mode']
            file_name = state['file_name']
            if mode == 'text' or mode == 'pdf':
                # Get the text content for summarization
                try:
                    collection = _get_chroma_collection()
                    text_docs = collection.get(
                        where={
                            "$and": [
                                {"source": file_name},
                                {"type": "text"}
                            ]
                        }
                    )
                    if text_docs and text_docs.get('documents'):
                        full_text = "\n".join(text_docs['documents'])
                        summary = summarize_transcript(full_text, summary_model=current_querying_model)
                        summaries.append(f"Summary for '{file_name}':\n{summary}\n")
                    else:
                        summaries.append(f"No text content found to summarize in '{file_name}'.\n")
                except Exception as e:
                    logger.error(f"Error summarizing {file_name}: {e}", exc_info=True)
                    summaries.append(f"Error summarizing {file_name}: {str(e)}\n")
            else:
                summaries.append(f"Note: '{file_name}' is an image file and cannot be summarized.\n")

        if summaries:
            return "\n---\n".join(summaries)
        return "No text content found to summarize in any of the processed files."

    # 2. Regular Query - Try to find relevant content across all files
    context_parts = []
    
    # First, try image-based responses for image/PDF files
    for file_path, state in processed_files.items():
        mode = state['processing_mode']
        file_name = state['file_name']
        
        if mode in ['image', 'pdf']:
            try:
                # Search for relevant image descriptions
                image_query_embed_func = OllamaEmbeddings(model=current_image_embedding_model) # Use image model
                query_embedding = image_query_embed_func.embed_query(query)
                collection = _get_chroma_collection()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=IMAGE_RETRIEVER_K,
                    where={
                        "$and": [
                            {"type": {"$eq": "image_description"}},
                            {"source": {"$eq": file_name}}
                        ]
                    },
                    include=['metadatas', 'distances']
                )

                if results and results.get('ids') and results['ids'][0]:
                    # Process the best match for each file
                    best_match_index = 0
                    if results.get('distances') and results['distances'][0]:
                        best_match_index = results['distances'][0].index(min(results['distances'][0]))
                    retrieved_metadata = results['metadatas'][0][best_match_index]

                    retrieved_image_path = retrieved_metadata.get('original_path')

                    if retrieved_image_path and os.path.exists(retrieved_image_path):
                        # Found a relevant image, query it
                        answer = query_llava_with_image(question=query, image_path=retrieved_image_path, llm_model=current_querying_model)
                        source_info = f" (from image in '{file_name}'"
                        if 'page_number' in retrieved_metadata:
                            source_info += f", page {retrieved_metadata['page_number']}"
                        source_info += ")"
                        context_parts.append(answer + source_info)

            except Exception as e:
                logger.error(f"Error processing image query for {file_name}: {e}", exc_info=True)

    # Then, try text-based responses
    try:
        # Collect text context from all files
        text_embed_func = OllamaEmbeddings(model=current_embedding_model)
        query_embedding = text_embed_func.embed_query(query)
        collection = _get_chroma_collection()

        all_text_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TEXT_RETRIEVER_K,
            where={"type": {"$eq": "text"}},  # Query all text chunks
            include=['documents', 'metadatas', 'distances']
        )

        if all_text_results and all_text_results.get('ids') and all_text_results['ids'][0]:
            for i, doc_content in enumerate(all_text_results['documents'][0]):
                meta = all_text_results['metadatas'][0][i]
                source = meta.get('source', 'Unknown source')
                page = meta.get('page_number')
                info = f"Source {i + 1} (from {source}"
                if page:
                    info += f", page {page}"
                info += ")"
                context_parts.append(f"{info}:\n{doc_content}")

    except Exception as e:
        logger.error(f"Error during text search: {e}", exc_info=True)
        if not context_parts:  # Only add error if we haven't found anything yet
            context_parts.append(f"Error searching text content: {str(e)}")

    if context_parts:
        # Combine all found context and generate final answer using current querying model
        context = "\n\n---\n\n".join(context_parts)
        try:
            final_answer = query_text_llm(query, context, current_querying_model)
            return final_answer
        except Exception as e:
            logger.error(f"Error generating final answer: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
    else:
        return "No relevant content found in any of the processed files for your query."
def check_ollama_models_sync():
    """Synchronous check for models (simplified version)."""
    try:
        logger.info("Skipping Ollama server check - using hardcoded models")
        return True
    except Exception as e:
        logger.warning("Error in model check (using hardcoded fallback)")
        return False
# --- Stdin/Stdout Communication ---

def send_response(data: Dict[str, Any]):
    """Send a JSON response to stdout for Electron to read."""
    try:
        # Ensure the data has all required fields
        if 'type' not in data:
            data['type'] = 'error'
            data['message'] = 'Response missing type field'
        
        response_json = json.dumps(data)
        print(response_json, flush=True)  # flush=True is critical for IPC
    except Exception as e:
        error_json = json.dumps({
            'type': 'error',
            'message': f'Error sending response: {str(e)}'
        })
        print(error_json, flush=True)

# --- Stdin/Stdout Communication ---
def main_loop():
    """Main loop to handle requests from Electron."""
    logger.info("Python backend started")

    try:
        for line in sys.stdin:
            try:
                command_data = json.loads(line)
                command = command_data.get('command')
                request_id = command_data.get('request_id', 'unknown')

                logger.info(f"Received command: {command} (request_id: {request_id})")

                if command == 'get_models':
                    models = get_available_models()
                    send_response({
                        'type': 'models_list',
                        'models': models,
                        'request_id': request_id,
                        'message': 'Retrieved available models'
                    })
                elif command == 'set_llm_model':
                    model_name = command_data.get('model_name')
                    if not model_name:
                        send_response({
                            'type': 'error',
                            'message': 'No model name provided',
                            'request_id': request_id
                        })
                    else:
                        result = set_llm_model(model_name)
                        send_response({
                            'type': 'model_set',
                            'message': result,
                            'request_id': request_id
                        })
                elif command == 'set_model':
                    model_name = command_data.get('model_name')
                    if not model_name:
                        send_response({
                            'type': 'error',
                            'message': 'No model name provided',
                            'request_id': request_id
                        })
                    else:
                        # set_model is an alias for set_querying_model
                        result = set_querying_model(model_name)
                        send_response({
                            'type': 'model_set',
                            'message': result,
                            'request_id': request_id
                        })
                elif command == 'set_embedding_model':
                    model_name = command_data.get('model_name')
                    if not model_name:
                        send_response({
                            'type': 'error',
                            'message': 'No model name provided',
                            'request_id': request_id
                        })
                    else:
                        result = set_embedding_model(model_name)
                        send_response({
                            'type': 'model_set',
                            'message': result,
                            'request_id': request_id
                        })
                elif command == 'set_image_embedding_model':
                    model_name = command_data.get('model_name')
                    if not model_name:
                        send_response({
                            'type': 'error',
                            'message': 'No model name provided',
                            'request_id': request_id
                        })
                    else:
                        result = set_image_embedding_model(model_name)
                        send_response({
                            'type': 'model_set',
                            'message': result,
                            'request_id': request_id
                        })
                elif command == 'set_querying_model':
                    model_name = command_data.get('model_name')
                    if not model_name:
                        send_response({
                            'type': 'error',
                            'message': 'No model name provided',
                            'request_id': request_id
                        })
                    else:
                        result = set_querying_model(model_name)
                        send_response({
                            'type': 'model_set',
                            'message': result,
                            'request_id': request_id
                        })
                elif command == 'query':
                    query_text = command_data.get('query_text')
                    if not query_text:
                        send_response({
                            'type': 'error',
                            'message': 'No query text provided',
                            'request_id': request_id
                        })
                        continue

                    try:
                        answer = _process_response_internal(query_text)
                        send_response({
                            'type': 'query_result',
                            'answer': answer,
                            'request_id': request_id
                        })
                    except Exception as e:
                        logger.error(f"Error processing query: {e}", exc_info=True)
                        send_response({
                            'type': 'error',
                            'message': f'Query processing error: {str(e)}',
                            'request_id': request_id
                        })
                elif command == 'process':
                    file_path = command_data.get('file_path')
                    if not file_path:
                        send_response({
                            'type': 'error',
                            'message': 'No file path provided',
                            'request_id': request_id
                        })
                        continue

                    # Process the document
                    try:
                        processing_mode, status = _process_document_internal(file_path)
                        if processing_mode is not None:
                            # Store the state for this file
                            processed_files[file_path] = {
                                'processing_mode': processing_mode,
                                'file_name': os.path.basename(file_path)
                            }
                            send_response({
                                'type': 'process_complete',
                                'status': status,
                                'mode': processing_mode,
                                'request_id': request_id
                            })
                        else:
                            send_response({
                                'type': 'error',
                                'message': f'Processing failed: {status}',
                                'request_id': request_id
                            })
                    except Exception as e:
                        logger.error(f"Error processing file: {e}", exc_info=True)
                        send_response({
                            'type': 'error',
                            'message': f'Processing error: {str(e)}',
                            'request_id': request_id
                        })
                else:
                    logger.warning(f"Unknown command received: {command}")
                    send_response({
                        'type': 'error',
                        'message': f'Unknown command: {command}',
                        'request_id': request_id
                    })

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse command JSON: {e}")
                send_response({
                    'type': 'error',
                    'message': f'Invalid JSON: {str(e)}',
                    'request_id': 'unknown'
                })
            except Exception as e:
                logger.error(f"Error processing command: {e}", exc_info=True)
                send_response({
                    'type': 'error',
                    'message': f'Command processing error: {str(e)}',
                    'request_id': 'unknown'
                })

    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main_loop()
