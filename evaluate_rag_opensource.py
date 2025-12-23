#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Evaluation for Open-Source Embedding Models
- Top-K retrieval metrikleri (Top-1/3/5, latency, throughput)
- (Opsiyonel) RAGAS: context_precision, context_recall, faithfulness
- Sorgu zenginleştirme: ürün başlığı + ASIN'i embedding'e kat ( --use-product-title )
"""

import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from tqdm import tqdm

# RAGAS (opsiyonel)
try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness  # answer_relevancy bilinçli olarak dahil değil
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    RAGAS_AVAILABLE = True
except Exception as e:
    RAGAS_AVAILABLE = False
    logging.warning(
        f"RAGAS not available. Install with: pip install ragas langchain-openai. Error: {e}"
    )

# ─────────── LOGGING ─────────── #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("opensource_evaluator")

# ─────────── ENV ─────────── #
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # RAGAS LLM için
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-3.5-haiku"

# Embedding side
from rag_loader_opensource import MODEL_CONFIGS, OpenSourceEmbeddings


# ─────────── DATA CLASSES ─────────── #
@dataclass
class QueryResult:
    query_id: str
    query_text: str
    ground_truth: str
    retrieved_contexts: List[str]
    retrieved_scores: List[float]
    top_k_hit: bool
    rank_of_ground_truth: Optional[int]
    avg_retrieval_score: float
    latency_ms: float


@dataclass
class ModelMetrics:
    model_name: str
    collection_name: str

    # Top-K
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float

    # RAGAS (opsiyonel)
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None  # Bilinçli boş bırakılabilir

    # Performans
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0

    # Ek bilgiler
    vector_dimension: int = 0
    total_queries: int = 0


# ─────────── UTILS ─────────── #
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception as e:
        logger.warning(f"Error calculating cosine similarity: {e}")
        return 0.0


# ─────────── EVALUATOR ─────────── #
class RAGEvaluator:
    """
    Retrieval odaklı değerlendirme.
    - Sorguyu (opsiyonel) ürün başlığı + ASIN ile zenginleştirir.
    - Doğru eşleşmeyi önce payload'taki parent_asin ile arar; bulunamazsa
      ground_truth ↔ context benzerliğine (eşik) fallback yapar.
    """

    def __init__(
        self,
        model_name: str,
        collection_name: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        use_product_title: bool = False
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_product_title = use_product_title

        # Embedding model
        config = MODEL_CONFIGS[model_name]
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.embeddings_model = OpenSourceEmbeddings(config, device=device)

        # Qdrant
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60.0
        )

        # dimension (rapor için)
        self.vector_dimension = getattr(self.embeddings_model, "dimension", 0)

        logger.info(f"Initialized evaluator for {model_name}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Top-K: {top_k}")
        logger.info(f"Use product title in queries: {self.use_product_title}")

    def _enrich_query(self, query_obj: Dict[str, Any]) -> str:
        """Sorguyu ürün başlığı/ASIN ile zenginleştir (opsiyonel)."""
        q = query_obj.get("question", "") or query_obj.get("query_text", "")
        if not self.use_product_title:
            return q
        title = (query_obj.get("product_title") or "").strip()
        asin = (query_obj.get("parent_asin") or "").strip()
        parts = [q]
        if title:
            parts.append(f"[PRODUCT_TITLE: {title}]")
        if asin:
            parts.append(f"[ASIN: {asin}]")
        return " ".join(parts)

    def _search_qdrant(self, enriched_query: str) -> Tuple[List[str], List[float], float, list]:
        """Qdrant search; contexts, scores, latency (ms), raw results döner."""
        start = time.perf_counter()
        q_emb = self.embeddings_model.embed_query(enriched_query)
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=q_emb,
            limit=self.top_k,
            with_payload=True
        )
        latency_ms = (time.perf_counter() - start) * 1000

        ctxs, scores = [], []
        for r in results:
            ctxs.append(r.payload.get("page_content", ""))
            scores.append(float(r.score))
        return ctxs, scores, latency_ms, results

    def evaluate_single_query(self, query_obj: Dict[str, Any]) -> Optional[QueryResult]:
        query_text = query_obj.get("question", "") or query_obj.get("query_text", "")
        gt_answer = query_obj.get("answer", "") or query_obj.get("ground_truth", "")
        qid = query_obj.get("id", str(hash(query_text)))
        gt_asin = query_obj.get("parent_asin")

        if not query_text or not gt_answer:
            logger.warning(f"Skipping invalid query: {qid}")
            return None

        # Enriched query → retrieve
        enriched = self._enrich_query(query_obj)
        contexts, scores, latency_ms, raw = self._search_qdrant(enriched)

        if not raw:
            return QueryResult(
                query_id=qid,
                query_text=query_text,
                ground_truth=gt_answer,
                retrieved_contexts=[],
                retrieved_scores=[],
                top_k_hit=False,
                rank_of_ground_truth=None,
                avg_retrieval_score=0.0,
                latency_ms=latency_ms
            )

        # 1) ASIN ile tam eşleşme (tercihli)
        rank_of_gt = None
        top_k_hit = False
        if gt_asin:
            for rank, r in enumerate(raw, start=1):
                res_asin = r.payload.get("review_parent_asin")
                if res_asin and res_asin == gt_asin:
                    rank_of_gt = rank
                    top_k_hit = True
                    break

        # 2) Fallback: semantik benzerlik (GT answer ↔ context)
        if not top_k_hit:
            try:
                gt_emb = self.embeddings_model.embed_query(gt_answer)
                best_sim, best_rank = 0.0, None
                for rank, ctx in enumerate(contexts, start=1):
                    ctx_emb = self.embeddings_model.embed_query(ctx)
                    sim = cosine_similarity(gt_emb, ctx_emb)
                    if sim > best_sim:
                        best_sim, best_rank = sim, rank
                if best_sim > self.similarity_threshold:
                    rank_of_gt = best_rank
                    top_k_hit = True
            except Exception as e:
                logger.warning(f"Similarity fallback failed for {qid}: {e}")

        return QueryResult(
            query_id=qid,
            query_text=query_text,
            ground_truth=gt_answer,
            retrieved_contexts=contexts,
            retrieved_scores=scores,
            top_k_hit=top_k_hit,
            rank_of_ground_truth=rank_of_gt,
            avg_retrieval_score=float(np.mean(scores)) if scores else 0.0,
            latency_ms=latency_ms
        )

    def evaluate_all_queries(self, queries: List[Dict[str, Any]]) -> Tuple[List[QueryResult], ModelMetrics]:
        logger.info(f"Evaluating {len(queries)} queries...")
        results: List[QueryResult] = []
        latencies: List[float] = []

        for q in tqdm(queries, desc="Evaluating queries"):
            r = self.evaluate_single_query(q)
            if r:
                results.append(r)
                latencies.append(r.latency_ms)

        total = len(results)
        top_1_hits = sum(1 for r in results if r.rank_of_ground_truth == 1)
        top_3_hits = sum(1 for r in results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 3)
        top_5_hits = sum(1 for r in results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 5)

        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        qps = (1000.0 / avg_lat) if avg_lat > 0 else 0.0

        metrics = ModelMetrics(
            model_name=self.model_name,
            collection_name=self.collection_name,
            top_1_accuracy=(top_1_hits / total) if total else 0.0,
            top_3_accuracy=(top_3_hits / total) if total else 0.0,
            top_5_accuracy=(top_5_hits / total) if total else 0.0,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            throughput_qps=qps,
            vector_dimension=self.vector_dimension,
            total_queries=total
        )
        return results, metrics

    def calculate_ragas_metrics(self, results: List[QueryResult]) -> Dict[str, float]:
        """
        RAGAS metrikleri (cevap üretimi olmadan):
        - context_precision
        - context_recall
        - faithfulness
        NOT: answer_relevancy anlamlı değil (model cevabı yok); bilinçli olarak hesaplanmıyor.
        """
        if not RAGAS_AVAILABLE or not OPENROUTER_API_KEY:
            if not RAGAS_AVAILABLE:
                logger.info("RAGAS not available → skipping LLM-based metrics.")
            elif not OPENROUTER_API_KEY:
                logger.info("OPENROUTER_API_KEY missing → skipping LLM-based metrics.")
            return {}

        logger.info(f"🔍 Calculating RAGAS metrics using {OPENROUTER_MODEL} (no answer_relevancy).")

        data = {
            "question": [],
            "answer": [],       # burada ground-truth'u koyuyoruz (model cevabı yok)
            "contexts": [],
            "ground_truth": []
        }
        for r in results:
            if r.retrieved_contexts:
                data["question"].append(r.query_text)
                data["answer"].append(r.ground_truth)
                data["contexts"].append(r.retrieved_contexts)
                data["ground_truth"].append(r.ground_truth)

        if not data["question"]:
            logger.warning("No valid samples for RAGAS.")
            return {}

        # RAGAS bazı backend'lerde OPENAI_API_KEY okur → OpenRouter anahtarını ona yansıt
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY

        # OpenRouter-LangChain LLM
        try:
            llm = ChatOpenAI(
                model=OPENROUTER_MODEL,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                temperature=0.0,
                max_tokens=1024,
                timeout=60.0
            )
        except Exception as e:
            logger.warning(f"LLM init failed with {OPENROUTER_MODEL}: {e} → fallback to openai/gpt-4o-mini")
            llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                temperature=0.0,
                max_tokens=1024,
                timeout=60.0
            )

        dataset = Dataset.from_dict(data)
        logger.info(f"✓ RAGAS dataset ready: {len(dataset)} samples")

        # Sadece context metrikleri (cevap üretimi olmadığından answer_relevancy'yi özellikle dışarıda bırakıyoruz)
        metrics_to_use = [context_precision, context_recall, faithfulness]
        logger.info("🚀 Running RAGAS (metrics: context_precision, context_recall, faithfulness)")

        try:
            ragas_result = evaluate(dataset, metrics=metrics_to_use, llm=llm, raise_exceptions=False)
            # RAGAS 0.3+ → pandas
            if hasattr(ragas_result, "to_pandas"):
                df = ragas_result.to_pandas()
                out = {
                    "context_precision": float(df.get("context_precision", []).mean()) if "context_precision" in df else 0.0,
                    "context_recall": float(df.get("context_recall", []).mean()) if "context_recall" in df else 0.0,
                    "faithfulness": float(df.get("faithfulness", []).mean()) if "faithfulness" in df else 0.0,
                }
            else:
                # eski sürümler için kaba fallback
                out = {
                    "context_precision": float(getattr(ragas_result, "context_precision", 0.0)),
                    "context_recall": float(getattr(ragas_result, "context_recall", 0.0)),
                    "faithfulness": float(getattr(ragas_result, "faithfulness", 0.0)),
                }

            logger.info("✓ RAGAS metrics calculated successfully:")
            for k, v in out.items():
                logger.info(f"  • {k}: {v:.4f}")
            return out

        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}")
            logger.info("Proceeding without LLM-based metrics.")
            return {}


# ─────────── CLI ─────────── #
def main():
    parser = argparse.ArgumentParser(description="Evaluate open-source embedding models (retrieval-focused)")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--category", type=str, default="Health_and_Personal_Care")
    parser.add_argument("--queries_file", type=str, default="benchmark_queries_v2.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Fallback similarity threshold")
    parser.add_argument("--use-product-title", action="store_true", help="Embed query with [PRODUCT_TITLE]/[ASIN] hints")
    # Bilinçli olarak answer_relevancy bayrağı kaldırıldı; cevap üretimi yokken anlamlı değil.

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Soruları yükle
    logger.info(f"Loading queries from: {args.queries_file}")
    try:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        logger.info(f"✓ Loaded {len(queries)} queries")
    except FileNotFoundError:
        logger.error(f"Queries file not found: {args.queries_file}")
        return

    collection_name = f"{args.category}_{args.model}"
    evaluator = RAGEvaluator(
        model_name=args.model,
        collection_name=collection_name,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        use_product_title=args.use_product_title
    )

    logger.info("\n" + "=" * 60)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 60)

    t0 = time.time()
    results, metrics = evaluator.evaluate_all_queries(queries)

    # RAGAS (cevap üretimi olmadan → answer_relevancy yok)
    ragas = evaluator.calculate_ragas_metrics(results)
    if ragas:
        metrics.context_precision = ragas.get("context_precision")
        metrics.context_recall = ragas.get("context_recall")
        metrics.faithfulness = ragas.get("faithfulness")
        metrics.answer_relevancy = None  # hesaplamıyoruz; 0.0 görünmesin diye None

    duration = time.time() - t0

    # Kaydet: sonuçlar
    results_file = os.path.join(args.output_dir, f"results_{args.model}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Detailed results saved to: {results_file}")

    # Kaydet: metrikler
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.model}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Metrics saved to: {metrics_file}")

    # Özet
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total queries: {metrics.total_queries}")

    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"  Top-1 Accuracy: {metrics.top_1_accuracy:.2%}")
    logger.info(f"  Top-3 Accuracy: {metrics.top_3_accuracy:.2%}")
    logger.info(f"  Top-5 Accuracy: {metrics.top_5_accuracy:.2%}")

    if ragas:
        logger.info(f"\nRAGAS Metrics:")
        logger.info(f"  Context Precision: {metrics.context_precision:.4f}")
        logger.info(f"  Context Recall: {metrics.context_recall:.4f}")
        logger.info(f"  Faithfulness: {metrics.faithfulness:.4f}")
        logger.info("  Answer Relevancy: (skipped — no model-generated answers)")

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Avg Latency: {metrics.avg_latency_ms:.2f}ms")
    logger.info(f"  P95 Latency: {metrics.p95_latency_ms:.2f}ms")
    logger.info(f"  P99 Latency: {metrics.p99_latency_ms:.2f}ms")
    logger.info(f"  Throughput: {metrics.throughput_qps:.2f} QPS")

    logger.info(f"\nModel Info:")
    logger.info(f"  Vector Dimension: {metrics.vector_dimension}")
    logger.info(f"  Evaluation Duration: {duration:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
