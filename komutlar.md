Elbette, `rag_loader_opensource.py` dosyanızdaki (sizin sağladığınız) `MODEL_CONFIGS` listesinde bulunan **tüm modeller** için eksiksiz yükleme (load) ve değerlendirme (evaluate) komut listesi aşağıdadır.

Komutlar, sizin de kullandığınız standart parametrelere (örn: `Health_and_Personal_Care`, `max_reviews 20000`, `benchmark_queries_v2.json`) dayanmaktadır.

-----

### 🚀 Hızlı Modeller (Önce Bunları Deneyin)

Bu modeller Mac M4 (MPS) üzerinde hızlı çalışmalıdır (genellikle \< 30 dakika).

#### `bge_m3`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model bge_m3 --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model bge_m3 --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `jina_v3`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model jina_v3 --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model jina_v3 --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `e5_large_instruct`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model e5_large_instruct --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model e5_large_instruct --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `qwen3_0_6b`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model qwen3_0_6b --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model qwen3_0_6b --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `snowflake_arctic` (Bu 'l' versiyonudur)

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model snowflake_arctic --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model snowflake_arctic --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `e5_base_instruct`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model e5_base_instruct --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model e5_base_instruct --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `snowflake_arctic_medium`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model snowflake_arctic_medium --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model snowflake_arctic_medium --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `gte_multilingual_base`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model gte_multilingual_base --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model gte_multilingual_base --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `nomic_embed_v1_5`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model nomic_embed_v1_5 --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model nomic_embed_v1_5 --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `e5_small`

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model e5_small --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model e5_small --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

-----

### 🐢 Yavaş Modeller (1B+ Parametre)

Bu modellerin Mac M4'ünüzde çalışması **saatler sürebilir**.

#### `gte_qwen2_1_5b` (1.5B)

(Bu, `use_fp16` ve `use_cache` düzeltmelerinizi içerir.)

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model gte_qwen2_1_5b --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model gte_qwen2_1_5b --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

-----

### 🚨 Çok Yavaş Modeller (7B Parametre)

Bu 7B modellerin Mac M4'ünüzde çalışması **günler sürebilir**. (Önce `max_reviews 100` gibi küçük bir sayı ile test etmeniz şiddetle tavsiye edilir.)

#### `sfr_mistral` (7B)

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model sfr_mistral --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model sfr_mistral --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `gritlm_7b` (7B)

  * **Load:**
    ```bash
    python rag_loader_opensource.py --model gritlm_7b --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model gritlm_7b --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```

#### `e5_mistral_7b` (7B)

  * **Load:**
    ```bash
    python rag_logo_opensource.py --model e5_mistral_7b --category Health_and_Personal_Care --max_reviews 20000
    ```
  * **Evaluate:**
    ```bash
    python evaluate_rag_opensource.py --model e5_mistral_7b --category Health_and_Personal_Care --queries_file benchmark_queries_v2.json --top_k 5 --output_dir evaluation_results
    ```