#!/usr/bin/env python3
"""
RAG Loader for Open-Source Embedding Models
Supports HuggingFace models via sentence-transformers
Optimized for:
  - Mac M4 with MPS (Metal Performance Shaders)
  - RunPod/Cloud with CUDA + INT8 Quantization
"""

import os
import json
import uuid
import time
import logging
import argparse
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────── LOGGING SETUP ─────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("opensource_loader")

# ─────────────────────────── ENV & CONFIG ───────────────────────────── #
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ─────────────────────────── DEVICE DETECTION ───────────────────────── #
def get_device() -> str:
    """Detect best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 CUDA available: {device_name}")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("🍎 MPS (Apple Silicon) available")
        return "mps"
    else:
        logger.info("💻 Using CPU")
        return "cpu"


# ─────────────────────────── MODEL CONFIGS ──────────────────────────── #
MODEL_CONFIGS = {
    # ═══════════════════════════════════════════════════════════════════
    # KÜÇÜK MODELLER (< 500M parameters)
    # ═══════════════════════════════════════════════════════════════════
    "e5_small": {
        "hf_model": "intfloat/multilingual-e5-small",
        "dimension": 384,
        "max_length": 512,
        "batch_size": 32,
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "e5_base_instruct": {
        "hf_model": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "e5_large_instruct": {
        "hf_model": "intfloat/multilingual-e5-large-instruct",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "all_minilm_l6_v2": {
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_length": 256,
        "batch_size": 32,
        "prefix": ""
    },
    "mpnet_base_v2": {
        "hf_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "prefix": ""
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # ORTA BOY MODELLER (500M - 1B parameters)
    # ═══════════════════════════════════════════════════════════════════
    "bge_m3": {
        "hf_model": "BAAI/bge-m3",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "prefix": ""
    },
    "gte_multilingual_base": {
        "hf_model": "Alibaba-NLP/gte-multilingual-base",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "prefix": "",
        "trust_remote_code": True
    },
    "jina_v3": {
        "hf_model": "jinaai/jina-embeddings-v3",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "prefix": "",
        "trust_remote_code": True
    },
    "nomic_embed_v1_5": {
        "hf_model": "nomic-ai/nomic-embed-text-v1.5",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "trust_remote_code": True
    },
    "snowflake_arctic": {
        "hf_model": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "prefix": ""
    },
    "qwen3_0_6b": {
        "hf_model": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "prefix": ""
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # BÜYÜK MODELLER (1B - 2B parameters)
    # ═══════════════════════════════════════════════════════════════════
    "gte_qwen2_1_5b": {
        "hf_model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "dimension": 1536,
        "max_length": 512,
        "batch_size": 4,
        "query_prefix": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
        "use_fp16": True,
        "model_kwargs": {
            "use_cache": False
        }
    },
    "stella_1_5b_v5": {
        "hf_model": "dunzhang/stella_en_1.5B_v5",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 4,
        "query_prefix": "Instruct: Retrieve semantically similar text\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
        "use_fp16": True
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # ÇOK BÜYÜK MODELLER (7B+ parameters) - CUDA + INT8 ÖNERİLİR
    # ═══════════════════════════════════════════════════════════════════
    "e5_mistral_7b": {
        "hf_model": "intfloat/e5-mistral-7b-instruct",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 2,
        "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "doc_prefix": "",
        "load_in_8bit": True,
        "use_fp16": True
    },
    "gte_qwen2_7b": {
        "hf_model": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "dimension": 3584,
        "max_length": 512,
        "batch_size": 1,
        "query_prefix": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": True,
        "use_fp16": True
    },
    "nv_embed_v2": {
        "hf_model": "nvidia/NV-Embed-v2",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 1,
        "query_prefix": "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": True,
        "use_fp16": True
    },
    "sfr_mistral": {
        "hf_model": "Salesforce/SFR-Embedding-Mistral",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 2,
        "query_prefix": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
        "doc_prefix": "",
        "load_in_8bit": True,
        "use_fp16": True
    },
    "bge_en_icl": {
        "hf_model": "BAAI/bge-en-icl",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 1,
        "prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": True,
        "use_fp16": True
    },
    "llm2vec_mistral_7b": {
        "hf_model": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 1,
        "prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": True,
        "use_fp16": True
    },
    "gritlm_7b": {
        "hf_model": "GritLM/GritLM-7B",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 1,
        "prefix": "",
        "load_in_8bit": True,
        "use_fp16": True
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # NVIDIA LLAMA EMBED NEMOTRON 8B - MTEB #1 (Ekim 2025)
    # Özel encode_query/encode_document metodları kullanır
    # ═══════════════════════════════════════════════════════════════════
    "llama_embed_nemotron_8b": {
        "hf_model": "nvidia/llama-embed-nemotron-8b",
        "dimension": 4096,
        "max_length": 4096,
        "batch_size": 1,
        "prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": True,
        "use_fp16": True,
        "use_native_encode": True,  # Özel flag: encode_query/encode_document kullan
        "model_kwargs": {
            "attn_implementation": "flash_attention_2",  # CUDA'da flash attention
            "torch_dtype": "bfloat16"
        },
        "tokenizer_kwargs": {
            "padding_side": "left"
        }
    },
}


# ──────────────────────────── EMBEDDING MODEL CLASS ─────────────────── #
class OpenSourceEmbeddings:
    """Wrapper for open-source embedding models using sentence-transformers"""
    
    def __init__(self, model_config: Dict[str, Any], device: str = None):
        self.config = model_config
        self.device = device or get_device()
        
        logger.info(f"Loading model: {model_config['hf_model']}")
        logger.info(f"Device: {self.device}, Expected Dimension: {model_config['dimension']}")
        
        # Build model kwargs
        model_kwargs = {}
        
        # Handle trust_remote_code
        if model_config.get("trust_remote_code", False):
            model_kwargs["trust_remote_code"] = True
        
        # Handle tokenizer kwargs (e.g., padding_side for NVIDIA model)
        tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})
        if tokenizer_kwargs:
            model_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
            logger.info(f"Tokenizer kwargs: {tokenizer_kwargs}")
        
        # ═══════════════════════════════════════════════════════════════
        # INT8 QUANTIZATION (CUDA ONLY)
        # ═══════════════════════════════════════════════════════════════
        if model_config.get("load_in_8bit", False) and self.device == "cuda":
            logger.info("🔧 Attempting INT8 quantization (bitsandbytes)...")
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                if "model_kwargs" not in model_kwargs:
                    model_kwargs["model_kwargs"] = {}
                model_kwargs["model_kwargs"]["quantization_config"] = quantization_config
                model_kwargs["model_kwargs"]["device_map"] = "auto"
                logger.info("✓ INT8 quantization configured")
            except ImportError:
                logger.warning("⚠️ bitsandbytes not installed, falling back to FP16")
                if "model_kwargs" not in model_kwargs:
                    model_kwargs["model_kwargs"] = {}
                model_kwargs["model_kwargs"]["torch_dtype"] = torch.float16
        
        # ═══════════════════════════════════════════════════════════════
        # FP16 / BFLOAT16 HANDLING
        # ═══════════════════════════════════════════════════════════════
        elif model_config.get("use_fp16", False):
            if "model_kwargs" not in model_kwargs:
                model_kwargs["model_kwargs"] = {}
            
            # Custom model_kwargs (e.g., attn_implementation, torch_dtype)
            custom_model_kwargs = model_config.get("model_kwargs", {})
            if custom_model_kwargs:
                # Handle string dtype conversion
                if "torch_dtype" in custom_model_kwargs:
                    dtype_str = custom_model_kwargs["torch_dtype"]
                    if dtype_str == "bfloat16":
                        custom_model_kwargs["torch_dtype"] = torch.bfloat16
                    elif dtype_str == "float16":
                        custom_model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["model_kwargs"].update(custom_model_kwargs)
                logger.info(f"Custom model_kwargs applied: {list(custom_model_kwargs.keys())}")
            
            # Default to float16 if no dtype specified
            if "torch_dtype" not in model_kwargs.get("model_kwargs", {}):
                model_kwargs["model_kwargs"]["torch_dtype"] = torch.float16
            
            logger.info("Loading with FP16/BF16 precision")
        
        # Handle CUDA-only features on non-CUDA devices
        if self.device != "cuda":
            if model_config.get("load_in_8bit", False):
                logger.warning("⚠️ INT8 quantization requires CUDA - skipping")
            # Remove flash_attention_2 for non-CUDA
            if "model_kwargs" in model_kwargs:
                if model_kwargs["model_kwargs"].get("attn_implementation") == "flash_attention_2":
                    model_kwargs["model_kwargs"]["attn_implementation"] = "eager"
                    logger.info("Switched to eager attention (flash_attention_2 requires CUDA)")
        
        # ═══════════════════════════════════════════════════════════════
        # LOAD MODEL
        # ═══════════════════════════════════════════════════════════════
        try:
            self.model = SentenceTransformer(
                model_config["hf_model"],
                device=self.device,
                **model_kwargs
            )
            logger.info("✓ Model loaded successfully")
            
            # Check for native encode methods
            self.has_native_encode = (
                model_config.get("use_native_encode", False) and
                hasattr(self.model, 'encode_query') and
                hasattr(self.model, 'encode_document')
            )
            if self.has_native_encode:
                logger.info("✓ Model has native encode_query/encode_document methods")
            
        except TypeError as e:
            # Some models don't accept certain kwargs - retry without them
            if "unexpected keyword argument" in str(e):
                logger.warning(f"Model doesn't support some kwargs, retrying: {e}")
                model_kwargs_clean = {"trust_remote_code": model_config.get("trust_remote_code", False)}
                self.model = SentenceTransformer(
                    model_config["hf_model"],
                    device=self.device,
                    **model_kwargs_clean
                )
                logger.info("✓ Model loaded (with minimal kwargs)")
                self.has_native_encode = False
            else:
                raise
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # ═══════════════════════════════════════════════════════════════
        # POST-LOAD HOTFIXES
        # ═══════════════════════════════════════════════════════════════
        self._apply_hotfixes()
        
        # Set max sequence length
        try:
            self.model.max_seq_length = self.config.get("max_length", 512)
            logger.info(f"Max seq length set to: {self.model.max_seq_length}")
        except Exception as e:
            logger.warning(f"Could not set max_seq_length: {e}")
        
        # Get actual embedding dimension
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = model_config["dimension"]
        
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def _apply_hotfixes(self):
        """Apply model-specific hotfixes after loading"""
        hf_model = self.config["hf_model"].lower()
        
        try:
            # Access the underlying transformer model
            if hasattr(self.model, '_first_module'):
                transformer = self.model._first_module()
                if hasattr(transformer, 'auto_model'):
                    auto_model = transformer.auto_model
                else:
                    return
            elif hasattr(self.model, '__getitem__'):
                try:
                    transformer = self.model[0]
                    if hasattr(transformer, 'auto_model'):
                        auto_model = transformer.auto_model
                    else:
                        return
                except:
                    return
            else:
                return
            
            # Qwen2/Stella: use_cache fix
            if "qwen2" in hf_model or "stella" in hf_model:
                if hasattr(auto_model, 'config'):
                    auto_model.config.use_cache = False
                    logger.info("✓ HOTFIX: use_cache=False applied (Qwen2/Stella)")
            
            # Snowflake: xformers fix
            if "snowflake-arctic" in hf_model:
                if hasattr(auto_model, 'config'):
                    auto_model.config.use_memory_efficient_attention = False
                    logger.info("✓ HOTFIX: use_memory_efficient_attention=False applied (Snowflake)")
                    
        except Exception as e:
            logger.debug(f"Hotfix not applicable: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not texts:
            return []
        
        # ═══════════════════════════════════════════════════════════════
        # NVIDIA model gibi özel encode_document metodu varsa kullan
        # ═══════════════════════════════════════════════════════════════
        if self.has_native_encode:
            logger.debug("Using native encode_document method")
            try:
                embeddings = self.model.encode_document(
                    texts,
                    batch_size=self.config["batch_size"],
                    show_progress_bar=False
                )
                # Convert to list format
                if hasattr(embeddings, 'cpu'):
                    embeddings = embeddings.cpu()
                if hasattr(embeddings, 'numpy'):
                    embeddings = embeddings.numpy()
                if hasattr(embeddings, 'tolist'):
                    return embeddings.tolist()
                return [list(emb) for emb in embeddings]
            except Exception as e:
                logger.warning(f"Native encode_document failed, falling back: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # Standard encode (prefix ile)
        # ═══════════════════════════════════════════════════════════════
        doc_prefix = self.config.get("doc_prefix", "")
        if doc_prefix:
            texts = [doc_prefix + text for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config["batch_size"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        
        # ═══════════════════════════════════════════════════════════════
        # NVIDIA model gibi özel encode_query metodu varsa kullan
        # ═══════════════════════════════════════════════════════════════
        if self.has_native_encode:
            logger.debug("Using native encode_query method")
            try:
                embedding = self.model.encode_query([query])
                # Handle different return shapes
                if hasattr(embedding, 'cpu'):
                    embedding = embedding.cpu()
                if hasattr(embedding, 'numpy'):
                    embedding = embedding.numpy()
                # If 2D, take first row
                if len(embedding.shape) > 1:
                    embedding = embedding[0]
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                return list(embedding)
            except Exception as e:
                logger.warning(f"Native encode_query failed, falling back: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # Standard encode (prefix ile)
        # ═══════════════════════════════════════════════════════════════
        query_prefix = self.config.get("query_prefix", "")
        if query_prefix:
            query = query_prefix + query
        
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding.tolist()


# ─────────────────────────── DATA LOADING ───────────────────────────── #
def load_and_index_data(
    model_name: str,
    category: str = "Health_and_Personal_Care",
    max_reviews: Optional[int] = None
):
    """
    Load reviews, create embeddings, and index them in Qdrant
    """
    # Get model config
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    config = MODEL_CONFIGS[model_name]
    
    # Initialize embedding model
    device = get_device()
    embeddings_model = OpenSourceEmbeddings(config, device=device)
    
    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        logger.info(f"✓ Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return
    
    # Create collection name
    collection_name = f"{category}_{model_name}"
    logger.info(f"Collection name: {collection_name}")
    
    # Create or recreate collection
    try:
        qdrant_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embeddings_model.dimension,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"✓ Created collection with dimension {embeddings_model.dimension}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return
    
    # Load metadata
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    metadata_file = os.path.join(data_dir, f"meta_{category}.jsonl")
    reviews_file = os.path.join(data_dir, f"{category}.jsonl")
    
    logger.info("Loading metadata...")
    metadata_map = {}
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                asin = item.get("parent_asin") or item.get("asin")
                if asin:
                    metadata_map[asin] = item
        logger.info(f"✓ Loaded {len(metadata_map)} metadata items")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_file}")
        return
    
    # Process reviews in batches
    logger.info("Processing reviews...")
    
    texts_to_embed = []
    payloads = []
    total_processed = 0
    total_skipped = 0
    
    embedding_batch_size = config["batch_size"]
    upsert_batch_size = 200
    points_batch = []
    
    start_time = time.time()
    
    try:
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading reviews", unit=" reviews"):
                if max_reviews and total_processed >= max_reviews:
                    logger.info(f"Reached max reviews limit: {max_reviews}")
                    break
                
                try:
                    review = json.loads(line)
                    
                    # Get review content
                    title = review.get("title", "")
                    text = review.get("text", "")
                    
                    if not text and not title:
                        total_skipped += 1
                        continue
                    
                    # Build enriched text
                    text_parts = []
                    if title:
                        text_parts.append(f"Review Title: {title}")
                    if text:
                        text_parts.append(f"Review: {text}")
                    
                    # Add product metadata if available
                    asin = review.get("parent_asin") or review.get("asin")
                    product_meta = metadata_map.get(asin) if asin else None
                    
                    if product_meta:
                        product_title = product_meta.get("title", "")
                        if product_title:
                            text_parts.append(f"Product: {product_title}")
                    
                    full_text = "\n".join(text_parts)
                    
                    # Create payload
                    payload = {
                        "page_content": full_text,
                        "review_rating": review.get("rating"),
                        "review_title": title,
                        "review_text": text,
                        "review_parent_asin": asin,
                    }
                    
                    if product_meta:
                        payload["product_title"] = product_meta.get("title")
                    
                    texts_to_embed.append(full_text)
                    payloads.append(payload)
                    total_processed += 1
                    
                    # Process batch when full
                    if len(texts_to_embed) >= embedding_batch_size:
                        embeddings = embeddings_model.embed_documents(texts_to_embed)
                        
                        for emb, pl in zip(embeddings, payloads):
                            points_batch.append(
                                models.PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=emb,
                                    payload=pl
                                )
                            )
                        
                        texts_to_embed = []
                        payloads = []
                        
                        # Upsert to Qdrant when batch is full
                        if len(points_batch) >= upsert_batch_size:
                            qdrant_client.upsert(
                                collection_name=collection_name,
                                points=points_batch,
                                wait=True
                            )
                            logger.info(f"✓ Uploaded {len(points_batch)} points (Total: {total_processed})")
                            points_batch = []
                
                except Exception as e:
                    logger.warning(f"Error processing review: {e}")
                    total_skipped += 1
                    continue
        
        # Process remaining items
        if texts_to_embed:
            logger.info("Processing final batch...")
            embeddings = embeddings_model.embed_documents(texts_to_embed)
            
            for emb, pl in zip(embeddings, payloads):
                points_batch.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload=pl
                    )
                )
        
        # Final upsert
        if points_batch:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points_batch,
                wait=True
            )
            logger.info(f"✓ Uploaded final batch of {len(points_batch)} points")
    
    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        return
    
    duration = time.time() - start_time
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DATA LOADING COMPLETED")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total skipped: {total_skipped}")
    logger.info(f"Vector dimension: {embeddings_model.dimension}")
    logger.info(f"Duration: {duration:.2f}s ({total_processed/duration:.1f} reviews/s)")
    logger.info("="*60)


# ─────────────────────────── MAIN CLI ───────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load open-source embedding models into Qdrant"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name to use"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Health_and_Personal_Care",
        help="Dataset category"
    )
    parser.add_argument(
        "--max_reviews",
        type=int,
        default=None,
        help="Maximum number of reviews to process (default: all)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting data loading for model: {args.model}")
    
    load_and_index_data(
        model_name=args.model,
        category=args.category,
        max_reviews=args.max_reviews
    )