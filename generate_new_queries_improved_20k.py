#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate diverse benchmark queries from Amazon reviews (first-N only)

- Sadece ilk N review: --reviews_limit (default: 20000)
- Kalite gardları:
  * SADECE REVIEW metnini kullan (prompt ile)
  * Ürün başlığı x review anahtar kelime örtüşme filtresi
  * HTML/ASIN işaretlerini temizleme
  * Cevap ≤ 30 kelime (gerekirse kırp)
  * Ürün (ASIN) başına en fazla N soru

Çalıştırma örneği:
python generate_new_queries_improved_20k.py \
  --category Health_and_Personal_Care \
  --num_questions 100 \
  --output_file benchmark_queries_v2.json \
  --reviews_limit 20000 \
  --min_helpful_votes 2 \
  --min_review_length 120 \
  --llm_model anthropic/claude-3.5-sonnet
"""

import os
import re
import json
import html
import argparse
import logging
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from dotenv import load_dotenv

# LangChain (modern importlar)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ───────────────────────── LOGGING ───────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("query_generator")

# ───────────────────────── ENV ───────────────────────── #
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ───────────────────────── Yardımcılar ───────────────────────── #
STOPWORDS = {
    "the","a","an","and","or","for","of","to","is","are","with","on","in","this","that",
    "it","its","by","from","as","at","be","was","were","i","you","he","she","they","we",
    "your","my","our","their","but","so","if","than","then","too","very","also","not"
}

def clean_review_text(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    # <br> -> newline
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    # HTML taglerini çıkar (temel)
    t = re.sub(r"<[^>]+>", " ", t)
    # [[ASIN:...]] gibi referansları temizle
    t = re.sub(r"\[\[\s*ASIN:[^\]]+\]\]", " ", t, flags=re.I)
    # Whitespace sadeleştir
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize_alpha(s: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", s)]

def overlap_ok(title: str, text: str, min_common: int = 2) -> bool:
    tw = set(tokenize_alpha(title)) - STOPWORDS
    rw = set(tokenize_alpha(text)) - STOPWORDS
    common = tw & rw
    return len(common) >= min_common

def truncate_words(s: str, max_words: int = 30) -> str:
    words = s.strip().split()
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]).rstrip(",.;:") + "."

# ───────────────────────── Türler & Prompts ───────────────────────── #
class QuestionType(Enum):
    FACTUAL = auto()
    OPINION = auto()
    USAGE = auto()
    PROBLEM_SOLVING = auto()
    FEATURE = auto()

BASE_GUARDRAILS = (
    "RULES:\n"
    "1) USE ONLY the REVIEW text as evidence. The product title is for topical focus only.\n"
    "2) If a valid QA cannot be formed from the REVIEW alone, output the single word: SKIP.\n"
    "3) Answer must be concise (max 30 words). Use the same units as in REVIEW; if the unit looks suspicious, append '(unit as written by reviewer)'.\n"
    "4) Output strictly as JSON with keys: question, answer. No extra text.\n"
)

PROMPT_TEMPLATES: Dict[QuestionType, str] = {
    QuestionType.FACTUAL: f"""
You are creating a FACTUAL question answerable strictly from the REVIEW text.

Examples:
- "How long does the battery last on a single charge?"
- "What size is the product?"
- "How many items are included?"

CONTEXT
Product Title: {{product_title}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{BASE_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.OPINION: f"""
Create an OPINION question about quality/suitability/value, answerable strictly from the REVIEW.

Examples:
- "Is this suitable for sensitive skin?"
- "Is it worth the price?"

CONTEXT
Product Title: {{product_title}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{BASE_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.USAGE: f"""
Create a USAGE question about how/when to use, answerable strictly from the REVIEW.

Examples:
- "How should I apply this?"
- "When is the best time to use it?"

CONTEXT
Product Title: {{product_title}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{BASE_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.PROBLEM_SOLVING: f"""
Create a PROBLEM-SOLVING question (does it address a specific issue?), answerable strictly from the REVIEW.

Examples:
- "Does this help with dry skin?"
- "Does it reduce acne breakouts?"

CONTEXT
Product Title: {{product_title}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{BASE_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.FEATURE: f"""
Create a FEATURE question about characteristics/capabilities, answerable strictly from the REVIEW.

Examples:
- "Does this have a strong scent?"
- "What ingredients are mentioned?"

CONTEXT
Product Title: {{product_title}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{BASE_GUARDRAILS}

{{format_instructions}}
""",
}

# ───────────────────────── Çekirdek Üretim ───────────────────────── #
def generate_diverse_queries(
    llm: ChatOpenAI,
    reviews_file: str,
    metadata_map: Dict[str, Any],
    num_questions: int,
    min_helpful_votes: int = 5,
    min_review_length: int = 150,
    reviews_limit: int = 20000,
    min_overlap_words: int = 2,
    asin_max_per_product: int = 2,
) -> List[Dict[str, Any]]:
    parser = JsonOutputParser()
    questions_per_type = {qt: 0 for qt in QuestionType}
    target_per_type = max(1, num_questions // len(QuestionType))
    asin_counts: Dict[str, int] = defaultdict(int)

    generated: List[Dict[str, Any]] = []
    processed_reviews = 0
    skipped_low_quality = 0
    skipped_overlap = 0
    skipped_asin_quota = 0
    skipped_skip_token = 0
    skipped_parse = 0

    types_cycle = list(QuestionType) * (num_questions // len(QuestionType) + 2)

    logger.info("Starting diverse query generation.")
    logger.info(f"Target: {num_questions} total questions")
    logger.info(f"Target per type: ~{target_per_type} questions")
    logger.info(f"Min helpful votes: {min_helpful_votes}")
    logger.info(f"Min review length: {min_review_length} chars")
    logger.info(f"Reviews scan limit: first {reviews_limit} lines")
    logger.info(f"Min title-review overlap words: {min_overlap_words}")
    logger.info(f"ASIN max questions: {asin_max_per_product}")

    with open(reviews_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if line_idx > reviews_limit:
                logger.info(f"Reached reviews_limit={reviews_limit}, stopping scan.")
                break
            if len(generated) >= num_questions:
                break

            try:
                review = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error at line {line_idx}: {e}")
                continue

            processed_reviews += 1

            helpful = review.get("helpful_vote", 0)
            raw_text = review.get("text", "") or ""
            parent_asin = review.get("parent_asin")
            rating = review.get("rating", 0)
            if (
                helpful < min_helpful_votes
                or not parent_asin
                or len(raw_text) < min_review_length
            ):
                skipped_low_quality += 1
                continue

            # metadata & title
            meta = metadata_map.get(parent_asin)
            if not meta or not meta.get("title"):
                skipped_low_quality += 1
                continue
            title = meta["title"]

            # clean review text
            review_text = clean_review_text(raw_text)

            # simple overlap check (ürün-review uyumu)
            if not overlap_ok(title, review_text, min_common=min_overlap_words):
                skipped_overlap += 1
                continue

            # ASIN başına kota
            if asin_counts[parent_asin] >= asin_max_per_product:
                skipped_asin_quota += 1
                continue

            # Tür seçimi ve tavan
            qtype = types_cycle[len(generated)]
            if questions_per_type[qtype] >= (target_per_type + 2):
                # başka türe şans verelim (basit fallback)
                for alt in QuestionType:
                    if questions_per_type[alt] < (target_per_type + 2):
                        qtype = alt
                        break

            prompt_template = PROMPT_TEMPLATES[qtype]
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["product_title", "review_text", "rating"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = prompt | llm | parser

            logger.info(
                f"[{len(generated)+1}/{num_questions}] "
                f"Line #{line_idx} (Helpful: {helpful}, Type: {qtype.name})"
            )

            try:
                qa = chain.invoke(
                    {
                        "product_title": title,
                        "review_text": review_text,
                        "rating": rating,
                    }
                )
            except Exception as e:
                logger.warning(f"LLM call failed at line {line_idx}: {e}")
                continue

            # Beklenen: dict with question, answer
            if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                skipped_parse += 1
                continue

            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()

            # "SKIP" sinyali veya boş alanlar
            if not question or not answer or question.upper() == "SKIP" or answer.upper() == "SKIP":
                skipped_skip_token += 1
                continue

            # cevap uzunluk gardı (≤30 kelime)
            answer = truncate_words(answer, max_words=30)

            item = {
                "id": f"q_{len(generated)+1:03d}",
                "question": question,
                "answer": answer,
                "context": review_text,
                "parent_asin": parent_asin,
                "product_title": title,
                "review_helpful_vote": helpful,
                "review_rating": rating,
                "question_type": qtype.name,
            }
            generated.append(item)
            questions_per_type[qtype] += 1
            asin_counts[parent_asin] += 1
            logger.info(f"✓ Generated {qtype.name} question")

    # Özet
    logger.info("\n" + "=" * 60)
    logger.info("QUERY GENERATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total generated: {len(generated)}")
    logger.info(f"Reviews processed (within limit): {processed_reviews}")
    logger.info(f"Skipped (low quality): {skipped_low_quality}")
    logger.info(f"Skipped (overlap failed): {skipped_overlap}")
    logger.info(f"Skipped (ASIN quota): {skipped_asin_quota}")
    logger.info(f"Skipped (SKIP token/empty): {skipped_skip_token}")
    logger.info(f"Skipped (parse errors): {skipped_parse}")
    logger.info("By type:")
    for qt, c in questions_per_type.items():
        logger.info(f"  {qt.name:18s}: {c:3d}")
    logger.info("=" * 60)
    return generated

# ───────────────────────── CLI ───────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse benchmark queries from Amazon reviews (first-N only)."
    )
    parser.add_argument("--category", default="Health_and_Personal_Care")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--output_file", default="benchmark_queries_v2.json")
    parser.add_argument("--min_helpful_votes", type=int, default=5)
    parser.add_argument("--min_review_length", type=int, default=150)
    parser.add_argument("--llm_model", default="anthropic/claude-3.5-sonnet")
    parser.add_argument("--reviews_limit", type=int, default=20000,
                        help="Only consider the first N reviews from the file")
    parser.add_argument("--min_overlap_words", type=int, default=2,
                        help="Minimum word overlap between title and review")
    parser.add_argument("--asin_max_per_product", type=int, default=2,
                        help="Max number of questions per parent_asin")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set; set it in your environment.")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    metadata_file = os.path.join(data_dir, f"meta_{args.category}.jsonl")
    reviews_file = os.path.join(data_dir, f"{args.category}.jsonl")

    # Metadata yükle
    logger.info("Loading metadata.")
    asin_to_meta: Dict[str, Any] = {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                asin = item.get("parent_asin")
                if asin:
                    asin_to_meta[asin] = item
        logger.info(f"✓ Loaded {len(asin_to_meta)} metadata items")
    except FileNotFoundError:
        logger.fatal(f"Metadata file not found: {metadata_file}")
        raise SystemExit(1)

    # LLM başlat (OpenRouter/OpenAI-uyumlu)
    logger.info(f"Initializing LLM: {args.llm_model}")
    llm = ChatOpenAI(
        model=args.llm_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.25,
        timeout=60,
    )

    # Üretim
    new_queries = generate_diverse_queries(
        llm=llm,
        reviews_file=reviews_file,
        metadata_map=asin_to_meta,
        num_questions=args.num_questions,
        min_helpful_votes=args.min_helpful_votes,
        min_review_length=args.min_review_length,
        reviews_limit=args.reviews_limit,
        min_overlap_words=args.min_overlap_words,
        asin_max_per_product=args.asin_max_per_product,
    )

    # Kaydet
    if new_queries:
        logger.info(f"Saving {len(new_queries)} questions to '{args.output_file}'.")
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            json.dump(new_queries, f_out, indent=2, ensure_ascii=False)
        logger.info("✓ Benchmark file saved successfully")
        # Örnek yazdır
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLE QUESTIONS (first 3)")
        logger.info("=" * 60)
        for i, item in enumerate(new_queries[:3], start=1):
            logger.info(f"[{i}] Q: {item.get('question')}")
            logger.info(f"    A: {item.get('answer')}")
            logger.info(f"    ASIN: {item.get('parent_asin')} — {item.get('product_title')}")

if __name__ == "__main__":
    main()
