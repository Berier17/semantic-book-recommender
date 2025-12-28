# Multilingual Semantic Book Recommender

This project is an end-to-end **Natural Language Processing (NLP)** solution designed to provide personalized book recommendations across languages. By leveraging **Sentence Transformers (BERT)**, the model bridges the gap between English and Arabic literature, recommending books based on *semantic meaning* rather than just keyword matching.

The system ingests data from **Goodreads** (User History) and **Hugging Face** (Arabic E-Book Corpus) to build a unified vector space for literary discovery.

## 1. Problem Description

The goal of this project is to solve two specific problems in the recommendation space:
1.  **The Vocabulary Gap:** Standard algorithms (like TF-IDF) fail to link concepts that use different words (e.g., "Space Travel" vs. "Interstellar Journey").
2.  **The Language Barrier:** A user who reads English Sci-Fi might love Arabic Sci-Fi, but traditional models keep these datasets separate.

This solution implements a **Content-Based Filtering** engine that treats books as "dense vectors." By analyzing metadata (Title, Author, Description, Genres), the model calculates a **"User Taste Centroid"**—a mathematical representation of a user's unique preference—to predict their next favorite book with high precision.

## 2. Data Engineering & Strategy

My analysis focused on merging disparate datasets to create a "Global Library." I utilized the **"Soup Strategy"** to engineer features that capture the full context of a book.

**1. Data Sources:**
* **Goodreads Data:** Accepts a standard `goodreads_export.csv` file containing a user's reading history, ratings, and shelves.
* **Arabic Corpus:** Integrated the **`mohres/The_Arabic_E-Book_Corpus`** from Hugging Face to expand the library with thousands of Arabic titles.

**2. Feature Engineering (The "Soup"):**
To ensure the model understands context, I concatenated all relevant metadata into a single text string for each book.

| Feature | Source | Purpose |
| :--- | :--- | :--- |
| **Title** | Goodreads/HF | Core identity of the item. |
| **Author** | Goodreads/HF | Captures writing style preferences. |
| **Description** | Goodreads/HF | Provides the plot and thematic context. |
| **Genres** | Goodreads/HF | High-level categorization (e.g., "Sci-Fi", "Philosophy"). |

**3. Multilingual Unification:**
By merging these datasets into a single DataFrame (`master_df`), I enabled the model to perform **Cross-Lingual Search**—finding Arabic matches for English queries without needing translation.

## 3. Model Architecture & Selection

I initially considered **TF-IDF (Term Frequency-Inverse Document Frequency)** but rejected it in favor of **Sentence Transformers**.

**Why Transformers?**
* **TF-IDF (Sparse):** Creates a massive, mostly empty matrix. It cannot understand that "Sad" and "Melancholy" are related.
* **Transformers (Dense):** Converts text into a compact list of numbers (embeddings). It understands that "King - Man + Woman = Queen."

**The Model:** `paraphrase-multilingual-MiniLM-L12-v2`
* **Architecture:** BERT-based Transformer.
* **Dimensions:** 384-dimensional dense vectors.
* **Language Support:** 50+ languages (enabling the English-Arabic bridge).

## 4. Usage & Workflows

The system offers two distinct modes of operation depending on the user's needs:

### Mode 1: Single Book Search (Book-to-Book)
* **Function:** `get_recommendations(query, ...)`
* **Input:** A specific book title (e.g., *"The Saga of Arthur Bandini"*).
* **Process:** The model converts the query into a vector and finds the nearest neighbors in the shared vector space.
* **Output:** Returns top 5 semantically similar books (English or Arabic).

### Mode 2: Profile Personalization (Profile-to-Book)
* **Function:** `get_profile_recommendations(my_df, ...)`
* **Input:** Upload your personal `goodreads_export.csv`.
* **Process (The "Taste Centroid"):**
    1.  **Filter:** Selects books the user rated **4 stars or higher**.
    2.  **Vectorize:** Converts these specific books into embeddings.
    3.  **Average:** Calculates the **Mean Vector (Centroid)** of these books to represent the user's "Taste Profile."
    4.  **Search:** Finds unread books closest to this centroid.

## 5. Reproducibility

This project is fully reproducible using Python.

### Prerequisites:
You will need the following libraries:
```bash
pip install pandas numpy scikit-learn sentence-transformers datasets
