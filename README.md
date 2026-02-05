# Hybrid RAG System with Automated Evaluation

A Retrieval-Augmented Generation (RAG) system that combines dense vector retrieval, sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from 500 Wikipedia articles. The system includes an automated evaluation framework with 100 generated questions.

## Group 8

| Name | Email |
|------|-------|
| SELVA PANDIAN S | 2023AC05005@wilp.bits-pilani.ac.in |
| Shikhar Nigam | 2024AA05691@wilp.bits-pilani.ac.in |
| Suraj Anand | 2024aa05731@wilp.bits-pilani.ac.in |
| NEERUMALLA KAVITHA | 2024AA05879@wilp.bits-pilani.ac.in |
| Karan Sharma | 2024AB05145@wilp.bits-pilani.ac.in |

---

## System Overview

The system implements a complete RAG pipeline with the following components:

1. **Data Collection** - Collects 500 Wikipedia articles (200 fixed + 300 random)
2. **Preprocessing** - Chunks text into 200-400 token segments with 50-token overlap
3. **Dense Retrieval** - FAISS vector search using `all-MiniLM-L6-v2` embeddings
4. **Sparse Retrieval** - BM25 keyword-based search
5. **Hybrid Fusion** - Reciprocal Rank Fusion combining both methods
6. **Answer Generation** - Flan-T5-base model for response generation
7. **Evaluation** - Automated evaluation with MRR, NDCG@K, and BERTScore metrics

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum
- Internet connection for Wikipedia data collection

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment (Optional)

Create a `.env` file for HuggingFace token (speeds up model downloads):

```bash
echo "HF_TOKEN=your_huggingface_token" > .env
```

### Step 4: Run the Pipeline

**Option A: Automated Setup (Recommended)**

```bash
./run.sh
```

This script handles everything: data collection, preprocessing, indexing, and launches the UI.

**Option B: Manual Step-by-Step**

```bash
# 1. Collect Wikipedia data (takes 15-20 minutes)
python src/data_collection.py

# 2. Preprocess and chunk the text
python src/preprocessing.py

# 3. Build retrieval indices (FAISS + BM25)
python src/hybrid_retrieval.py

# 4. Launch the UI (choose one)
python ui/app.py           # Flask interface on http://localhost:5000
python ui/gradio_app.py    # Gradio interface on http://localhost:5001
```

### Step 5: Run Evaluation

```bash
python run_evaluation.py
```

This generates 100 test questions, runs them through the system, and creates an evaluation report at `reports/evaluation_report.html`.

---

## Project Structure

```
├── config.py                    # Central configuration file
├── requirements.txt             # Python dependencies
├── run.sh                       # Automated setup script
├── run_evaluation.py            # Main evaluation runner
│
├── src/                         # Core RAG implementation
│   ├── data_collection.py       # Wikipedia scraping (200 fixed + 300 random URLs)
│   ├── preprocessing.py         # Text chunking with overlap
│   ├── embeddings.py            # Dense retrieval with FAISS
│   ├── sparse_retrieval.py      # Sparse retrieval with BM25
│   ├── hybrid_retrieval.py      # RRF fusion of dense + sparse
│   └── llm_generation.py        # Answer generation with Flan-T5
│
├── evaluation/                  # Automated evaluation framework
│   ├── question_generation.py   # Generates 100 Q&A pairs from corpus
│   ├── metrics.py               # MRR, NDCG@K, BERTScore implementation
│   ├── pipeline.py              # Evaluation orchestration
│   ├── innovative_eval.py       # Ablation studies and error analysis
│   ├── report_generator.py      # HTML report generation
│   └── questions_dataset.json   # Generated evaluation questions
│
├── ui/                          # User interfaces
│   ├── app.py                   # Flask web interface
│   ├── gradio_app.py            # Gradio chat interface
│   ├── templates/               # HTML templates for Flask
│   └── static/                  # CSS and JS assets
│
├── data/                        # Data files (generated)
│   ├── fixed_urls.json          # 200 fixed Wikipedia URLs
│   ├── corpus.pkl               # Downloaded articles
│   ├── chunks.json              # Preprocessed text chunks
│   ├── faiss_index.bin          # Vector search index
│   ├── faiss_index_metadata.pkl # Index metadata
│   └── bm25_index.pkl           # BM25 keyword index
│
├── reports/                     # Evaluation outputs
│   ├── evaluation_report.html   # Main HTML report with visualizations
│   ├── results.json             # Raw evaluation results
│   └── results.csv              # Results in CSV format
│
├── docs/                        # Documentation
│   └── system_dataflow.png      # Architecture diagram
│
├── Dockerfile                   # Docker container setup
└── docker-compose.yml           # Docker Compose configuration
```

---

## Dataset

The system uses a hybrid dataset strategy:

- **Fixed URLs (200)**: Curated list stored in `data/fixed_urls.json` for reproducibility
- **Random URLs (300)**: Freshly sampled from Wikipedia API on each run for robustness

**Total Corpus**: 500 Wikipedia articles with minimum 200 words per article.

---

## Hybrid Retrieval

### Dense Retrieval
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings (384 dimensions)
- FAISS index for fast similarity search
- Returns top-K chunks by cosine similarity

### Sparse Retrieval
- BM25 algorithm for keyword-based ranking
- Handles exact term matching and TF-IDF weighting

### Reciprocal Rank Fusion (RRF)
Combines results from both methods using the formula:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where k=60 (as specified in assignment). This method ensures documents appearing in both result sets get higher scores.

---

## Answer Generation

The system uses `google/flan-t5-base` for answer generation:
- Top 5 chunks from RRF are concatenated as context
- Model generates natural language answers based on retrieved context
- Maximum generation length: 256 tokens

---

## Evaluation Metrics

### Mandatory Metric

| Metric | Description |
|--------|-------------|
| **MRR (Mean Reciprocal Rank)** | Measures how quickly the system finds the correct source URL. Calculated at URL level, not chunk level. |

### Custom Metrics

| Metric | Justification | Interpretation |
|--------|---------------|----------------|
| **NDCG@5** | Evaluates ranking quality by considering position of all relevant documents, not just the first. | 1.0 = perfect ranking, 0.7+ = good, <0.5 = poor |
| **BERTScore** | Measures semantic similarity using contextual embeddings, capturing meaning beyond exact word overlap. | >0.9 = excellent, 0.8-0.9 = good, <0.7 = weak |

---

## User Interface

### Flask Interface (`ui/app.py`)

```bash
python ui/app.py
```

Access at `http://localhost:5000`. Features:
- Query input with real-time search
- Display of generated answer with sources
- Shows dense/sparse/RRF scores for transparency
- Response time tracking

### Gradio Interface (`ui/gradio_app.py`)

```bash
python ui/gradio_app.py
```

Access at `http://localhost:5001`. Features:
- Chat-style interface
- Shareable public link option
- Example questions for quick testing

---

## Docker

Build and run using Docker Compose:

```bash
docker-compose up --build
```

Access the Flask UI at `http://localhost:5000`.

To stop:

```bash
docker-compose down
```

---

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIXED_URLS_COUNT` | 200 | Number of fixed Wikipedia URLs |
| `RANDOM_URLS_COUNT` | 300 | Number of random URLs per run |
| `MIN_CHUNK_TOKENS` | 200 | Minimum tokens per chunk |
| `MAX_CHUNK_TOKENS` | 400 | Maximum tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 50 | Token overlap between chunks |
| `DENSE_TOP_K` | 10 | Candidates from dense retrieval |
| `SPARSE_TOP_K` | 10 | Candidates from sparse retrieval |
| `RRF_K` | 60 | RRF smoothing constant |
| `FINAL_TOP_N` | 5 | Final chunks passed to LLM |
| `QUESTIONS_COUNT` | 100 | Number of evaluation questions |

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `corpus.pkl not found` | Run `python src/data_collection.py` |
| `chunks.json not found` | Run `python src/preprocessing.py` |
| `faiss_index.bin not found` | Run `python src/hybrid_retrieval.py` |
| Out of memory | Reduce `DENSE_TOP_K` and `SPARSE_TOP_K` in config.py |
| Slow data collection | Normal behavior due to Wikipedia rate limits (~15-20 min) |

---

## Dependencies

Core libraries used:

- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector similarity search
- `rank-bm25` - BM25 algorithm
- `transformers` - Flan-T5 model
- `flask` / `gradio` - Web interfaces
- `bert-score` - Answer evaluation
- `beautifulsoup4` - Wikipedia scraping
- `tiktoken` - Token counting

See `requirements.txt` for complete list.
