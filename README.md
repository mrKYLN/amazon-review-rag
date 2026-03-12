# 🛒 Amazon Review RAG - Semantic Search with LLM Summaries

A full-stack NLP project that lets you search 568,000 Amazon food reviews using natural language - in **any language** - and get AI-generated summaries in Turkish.

Built with FAISS vector search, sentence-transformers embeddings, Gemini LLM, and a Gradio web interface.

---

## ✨ Features

- 🔍 **Semantic search** - finds relevant reviews by meaning, not just keywords
- 🌐 **Multilingual queries** - type in Turkish, search in English automatically
- 🤖 **LLM summaries** - Gemini 2.5 Flash summarizes results in Turkish
- ⭐ **Filter by rating** - narrow results to 1★–5★ reviews
- 📦 **Filter by product** - search within a specific Amazon product ID
- ⚡ **Fast** - FAISS cosine similarity search over 10,000 vectors in milliseconds

---

## 🏗️ Architecture

```
User query (any language)
        │
        ▼
  Gemini translates
  query to English
        │
        ▼
sentence-transformers     →   384-dim vector
  (all-MiniLM-L6-v2)
        │
        ▼
  FAISS IndexFlatIP        →   Top-K similar reviews
  (cosine similarity)           + score/product filter
        │
        ▼
  Gemini 2.5 Flash         →   Turkish summary
  (RAG prompt)
        │
        ▼
    Gradio UI
```

---

## 📊 Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) — 568,454 reviews from Amazon spanning 10+ years.

- **Indexed:** 10,000 reviews (2,000 per rating, balanced sampling)
- **Fields used:** `ProductId`, `Score`, `Summary`, `Text`

> Download `Reviews.csv` from Kaggle and place it in the project root before running the notebook.

---

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/mrKYLN/amazon-review-rag.git
cd amazon-review-rag
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download the dataset**

Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it in the project root.

**4. Build the index**

Run all cells in `notebook.ipynb`. This will:
- Clean and sample the data
- Generate embeddings with `all-MiniLM-L6-v2`
- Build and save the FAISS index (`reviews.index`, `chunks.pkl`, `metadata.pkl`)

**5. Get a Gemini API key**

Create a free key at [Google AI Studio](https://aistudio.google.com/apikey).

**6. Run the app**
```bash
GEMINI_API_KEY=your_key_here python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## 🖥️ Screenshots

| Search | Results |
|--------|---------|
| Turkish query "çikolata" → auto-translated → English FAISS search → Turkish LLM summary | Filtered by 1★ reviews to find the most common complaints |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data cleaning and sampling |
| `sentence-transformers` | Text → vector embeddings |
| `faiss-cpu` | Fast vector similarity search |
| `google-genai` | Query translation + Turkish summaries |
| `gradio` | Web interface |

---

## 📁 Project Structure

```
amazon-review-rag/
├── notebook.ipynb     # Data prep, embedding, FAISS index
├── app.py             # Gradio web app
├── requirements.txt
└── README.md
```

---

## 💡 How RAG Works Here

**RAG (Retrieval-Augmented Generation)** means the LLM doesn't answer from memory - it answers based on real retrieved documents.

1. **Retrieve** - FAISS finds the most relevant reviews for your query
2. **Augment** - those reviews are added to the LLM prompt as context
3. **Generate** - Gemini writes a summary based only on those reviews

This makes answers grounded in real user opinions, not hallucinated.
