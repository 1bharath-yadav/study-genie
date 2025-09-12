"""
Pure functional file processing with LLM integration
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from functools import partial
from dataclasses import dataclass

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Retrieval and chains
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader
)

# For advanced document processing
from unstructured.partition.auto import partition
import google.generativeai as genai
import PIL.Image
import io

from app.llm.types import LLMRequest
from app.llm.providers import generate_text

logger = logging.getLogger(__name__)

# Pure constants - immutable configuration
@dataclass(frozen=True)
class ProcessingConfig:
    """Immutable configuration for document processing"""
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 6
    alpha: float = 0.75
    model_id: str = "gemini-2.0-flash-exp"
    embedding_model: str = "models/embedding-001"
    max_tokens: int = 4000
    min_text_threshold: int = 50

# Default configuration instance
DEFAULT_CONFIG = ProcessingConfig()

# Pure data structures
@dataclass(frozen=True)
class FileInfo:
    """Immutable file information"""
    path: str
    extension: str
    size: int

@dataclass(frozen=True)
class ProcessingResult:
    """Immutable processing result"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    error: Optional[str] = None

@dataclass(frozen=True)
class DocumentChunk:
    """Immutable document chunk"""
    content: str
    metadata: Dict[str, Any]
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain document"""
        return Document(page_content=self.content, metadata=self.metadata)

# Pure utility functions
def create_file_info(file_path: str) -> FileInfo:
    """Pure function to create file info"""
    path_obj = Path(file_path)
    return FileInfo(
        path=file_path,
        extension=path_obj.suffix.lower(),
        size=path_obj.stat().st_size if path_obj.exists() else 0
    )

def get_loader_factory(extension: str) -> Callable[[str], Any]:
    """Pure function that returns appropriate loader factory"""
    loader_map = {
        '.pdf': PyPDFLoader,
        '.txt': partial(TextLoader, encoding='utf-8'),
        '.png': UnstructuredImageLoader,
        '.jpg': UnstructuredImageLoader,
        '.jpeg': UnstructuredImageLoader,
        '.bmp': UnstructuredImageLoader,
        '.tiff': UnstructuredImageLoader,
        '.gif': UnstructuredImageLoader,
    }
    return loader_map.get(extension, UnstructuredFileLoader)

def create_ocr_prompt(file_extension: str) -> str:
    """Pure function to create OCR prompt based on file type"""
    if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
        return """Please extract all text content from this image. 
        Include any handwritten text, printed text, tables, diagrams with text, etc.
        Format the output as clean, readable text maintaining the original structure where possible.
        If there are mathematical formulas, describe them clearly.
        If there are diagrams, describe them briefly along with any text they contain."""
    
    return f"""Please analyze this document and extract all meaningful content.
    The file type is {file_extension}. Extract:
    - All text content
    - Structure and formatting
    - Any tables or data
    - Key information and concepts
    
    Format as clean, readable text."""

def create_enhanced_metadata(
    base_metadata: Dict[str, Any], 
    file_info: FileInfo, 
    content_length: int
) -> Dict[str, Any]:
    """Pure function to create enhanced metadata"""
    return {
        **base_metadata,
        'source_file': file_info.path,
        'file_type': file_info.extension,
        'original_length': content_length,
        'file_size': file_info.size
    }

def create_chunk_metadata(
    base_metadata: Dict[str, Any], 
    chunk_index: int, 
    total_chunks: int, 
    chunk_size: int
) -> Dict[str, Any]:
    """Pure function to create chunk metadata"""
    return {
        **base_metadata,
        'chunk_id': chunk_index,
        'chunk_size': chunk_size,
        'chunk_index': chunk_index,
        'total_chunks': total_chunks
    }

def is_content_sufficient(content: str, threshold: int = DEFAULT_CONFIG.min_text_threshold) -> bool:
    """Pure function to check if content is sufficient"""
    return len(content.strip()) >= threshold

def filter_non_empty_documents(documents: List[Document]) -> List[Document]:
    """Pure function to filter out empty documents"""
    return [doc for doc in documents if doc.page_content.strip()]

def sort_chunks_by_id(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Pure function to sort chunks by chunk_id"""
    return sorted(chunks, key=lambda x: x.metadata.get('chunk_id', 0))

def get_neighbor_chunk_ids(chunk_id: int, existing_ids: set, offsets: Optional[List[int]] = None) -> List[int]:
    """Pure function to get neighboring chunk IDs"""
    if offsets is None:
        offsets = [-2, -1, 1, 2]
    
    return [
        chunk_id + offset 
        for offset in offsets 
        if (chunk_id + offset) not in existing_ids and (chunk_id + offset) >= 0
    ]

# Impure I/O functions (clearly separated)
async def validate_api_key(api_key: str, config: ProcessingConfig = DEFAULT_CONFIG) -> bool:
    """Impure function to validate API key"""
    try:
        genai.configure(api_key=api_key)  # type: ignore
        test_model = genai.GenerativeModel(config.model_id)  # type: ignore
        await asyncio.to_thread(test_model.generate_content, "Hello")
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

async def extract_text_with_unstructured(file_path: str) -> ProcessingResult:
    """Impure function to extract text using unstructured"""
    try:
        elements = await asyncio.to_thread(partition, filename=file_path)
        content = "\n".join([str(element) for element in elements])
        return ProcessingResult(
            success=True,
            content=content,
            metadata={'extraction_method': 'unstructured'}
        )
    except Exception as e:
        return ProcessingResult(
            success=False,
            content="",
            metadata={'extraction_method': 'unstructured'},
            error=str(e)
        )

async def extract_text_with_llm_vision(
    file_info: FileInfo, 
    api_key: str, 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> ProcessingResult:
    """Impure function to extract text using LLM vision"""
    try:
        genai.configure(api_key=api_key)  # type: ignore
        model = genai.GenerativeModel(config.model_id)  # type: ignore
        
        with open(file_info.path, 'rb') as f:
            file_data = f.read()
        
        if file_info.extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
            image = PIL.Image.open(io.BytesIO(file_data))
            prompt = create_ocr_prompt(file_info.extension)
            response = await asyncio.to_thread(model.generate_content, [prompt, image])
            content = response.text if response.text else ""
        else:
            # Try unstructured first
            unstructured_result = await extract_text_with_unstructured(file_info.path)
            
            if unstructured_result.success and is_content_sufficient(unstructured_result.content):
                return unstructured_result
            
            # Fallback to LLM processing
            prompt = create_ocr_prompt(file_info.extension)
            request = LLMRequest(
                prompt=prompt,
                model_id=config.model_id,
                user_id="system",
                max_tokens=config.max_tokens
            )
            response = await generate_text(request, api_key)
            content = response.content
        
        return ProcessingResult(
            success=True,
            content=content,
            metadata={'extraction_method': 'llm_vision'}
        )
        
    except Exception as e:
        return ProcessingResult(
            success=False,
            content="",
            metadata={'extraction_method': 'llm_vision'},
            error=str(e)
        )

async def load_document_with_loader(file_info: FileInfo) -> ProcessingResult:
    """Impure function to load document using appropriate loader"""
    try:
        loader_factory = get_loader_factory(file_info.extension)
        loader = loader_factory(file_info.path)
        docs = await asyncio.to_thread(loader.load)
        
        # Combine all document content
        content = "\n\n".join([doc.page_content for doc in docs])
        base_metadata = docs[0].metadata if docs else {}
        
        return ProcessingResult(
            success=True,
            content=content,
            metadata={**base_metadata, 'extraction_method': 'loader'}
        )
        
    except Exception as e:
        return ProcessingResult(
            success=False,
            content="",
            metadata={'extraction_method': 'loader'},
            error=str(e)
        )

# Pure processing pipeline functions
async def process_single_file(
    file_path: str, 
    api_key: Optional[str] = None, 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> List[Document]:
    """Process a single file through the extraction pipeline"""
    file_info = create_file_info(file_path)
    
    # Try standard loader first
    loader_result = await load_document_with_loader(file_info)
    
    if loader_result.success and is_content_sufficient(loader_result.content):
        # Success with standard loader
        enhanced_metadata = create_enhanced_metadata(
            loader_result.metadata,
            file_info,
            len(loader_result.content)
        )
        return [Document(page_content=loader_result.content, metadata=enhanced_metadata)]
    
    # Fallback to LLM processing if API key available
    if api_key:
        llm_result = await extract_text_with_llm_vision(file_info, api_key, config)
        if llm_result.success:
            enhanced_metadata = create_enhanced_metadata(
                llm_result.metadata,
                file_info,
                len(llm_result.content)
            )
            return [Document(page_content=llm_result.content, metadata=enhanced_metadata)]
    
    # Create error document
    error_content = f"Could not extract text from {file_path}. File type: {file_info.extension}"
    error_metadata = create_enhanced_metadata(
        {'error': loader_result.error or 'Unknown error'},
        file_info,
        len(error_content)
    )
    
    return [Document(page_content=error_content, metadata=error_metadata)]

async def load_documents_from_files(
    file_paths: List[str], 
    api_key: Optional[str] = None, 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> List[Document]:
    """Pure functional document loading pipeline"""
    if not file_paths:
        return []
    
    # Process all files concurrently
    tasks = [
        process_single_file(file_path, api_key, config) 
        for file_path in file_paths
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten results and filter out exceptions
    all_documents: List[Document] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"File processing failed: {result}")
            continue
        # Type guard: result is now guaranteed to be List[Document]
        if isinstance(result, list):
            all_documents.extend(result)
    
    filtered_documents = filter_non_empty_documents(all_documents)
    
    if not filtered_documents:
        raise ValueError("No documents could be loaded from the provided files")
    
    logger.info(f"Successfully loaded {len(filtered_documents)} documents from {len(file_paths)} files")
    return filtered_documents

def create_text_splitter(config: ProcessingConfig = DEFAULT_CONFIG) -> RecursiveCharacterTextSplitter:
    """Pure function to create text splitter"""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        keep_separator=True
    )

def create_document_chunks(
    documents: List[Document], 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> List[DocumentChunk]:
    """Pure functional document chunking"""
    text_splitter = create_text_splitter(config)
    langchain_chunks = text_splitter.split_documents(documents)
    
    # Convert to immutable chunks with enhanced metadata
    chunks = []
    total_chunks = len(langchain_chunks)
    
    for idx, chunk in enumerate(langchain_chunks):
        enhanced_metadata = create_chunk_metadata(
            chunk.metadata,
            idx,
            total_chunks,
            len(chunk.page_content)
        )
        
        chunks.append(DocumentChunk(
            content=chunk.page_content,
            metadata=enhanced_metadata
        ))
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

async def perform_document_chunking(
    documents: List[Document], 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> List[Document]:
    """Wrapper function for backward compatibility"""
    chunks = create_document_chunks(documents, config)
    return [chunk.to_langchain_document() for chunk in chunks]

async def setup_vector_store_and_retriever(
    chunks: List[Document], 
    user_api_key: str, 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> Tuple[FAISS, EnsembleRetriever]:
    """Setup vector store and retriever with validation"""
    # Validate inputs
    if not user_api_key or not user_api_key.strip():
        raise ValueError("API key is required and cannot be empty")
    
    if not chunks:
        raise ValueError("No chunks provided for vector store creation")
    
    clean_api_key = user_api_key.strip()
    
    # Validate API key
    is_valid = await validate_api_key(clean_api_key, config)
    if not is_valid:
        raise ValueError("Invalid API key provided")
    
    # Create embeddings
    try:
        from pydantic import SecretStr
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=SecretStr(clean_api_key)
        )
        logger.info("Successfully initialized Google Generative AI embeddings")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise ValueError(f"Failed to initialize embeddings: {e}")
    
    # Create vector store
    try:
        vector_store = await asyncio.to_thread(FAISS.from_documents, chunks, embeddings)
        logger.info(f"Stored {len(chunks)} chunks in FAISS vector store")
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise ValueError(f"Failed to create vector store: {e}")
    
    # Setup retrievers
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.top_k}
    )
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.top_k
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[config.alpha, 1.0 - config.alpha]
    )
    
    logger.info("Successfully configured hybrid retrieval system")
    return vector_store, ensemble_retriever

def find_chunks_by_ids(chunks: List[Document], target_ids: List[int]) -> List[Document]:
    """Pure function to find chunks by their IDs"""
    id_set = set(target_ids)
    return [
        chunk for chunk in chunks 
        if chunk.metadata.get('chunk_id') in id_set
    ]

def enhance_retrieved_context(
    retrieved_docs: List[Document], 
    all_chunks: List[Document]
) -> List[Document]:
    """Pure functional context enhancement"""
    if not retrieved_docs or not all_chunks:
        return retrieved_docs
    
    # Get existing chunk IDs
    existing_ids = {doc.metadata.get('chunk_id', -1) for doc in retrieved_docs}
    existing_ids.discard(-1)  # Remove invalid IDs
    
    # Find all neighbor IDs
    all_neighbor_ids = []
    for doc in retrieved_docs:
        chunk_id = doc.metadata.get('chunk_id', -1)
        if chunk_id != -1:
            neighbor_ids = get_neighbor_chunk_ids(chunk_id, existing_ids)
            all_neighbor_ids.extend(neighbor_ids)
    
    # Remove duplicates and add to existing IDs
    unique_neighbor_ids = list(set(all_neighbor_ids))
    existing_ids.update(unique_neighbor_ids)
    
    # Find neighbor chunks
    neighbor_chunks = find_chunks_by_ids(all_chunks, unique_neighbor_ids)
    
    # Combine and sort
    enhanced_docs = list(retrieved_docs) + neighbor_chunks
    enhanced_docs.sort(key=lambda x: x.metadata.get('chunk_id', 0))
    
    logger.info(f"Enhanced context from {len(retrieved_docs)} to {len(enhanced_docs)} chunks")
    return enhanced_docs

def format_context_for_llm(documents: List[Document]) -> str:
    """Pure function to format context for LLM"""
    if not documents:
        return ""
    
    context_parts = []
    for idx, doc in enumerate(documents, 1):
        source = doc.metadata.get('source_file', 'Unknown')
        content = doc.page_content.strip()
        
        if content:  # Only include non-empty content
            formatted_section = f"""=== Document Section {idx} ===
Source: {source}
Content: {content}"""
            context_parts.append(formatted_section)
    
    return "\n\n".join(context_parts)

# Export main functions for backward compatibility
__all__ = [
    'ProcessingConfig',
    'DEFAULT_CONFIG',
    'load_documents_from_files',
    'perform_document_chunking', 
    'setup_vector_store_and_retriever',
    'enhance_retrieved_context',
    'format_context_for_llm',
    'extract_text_with_llm_vision'
]
