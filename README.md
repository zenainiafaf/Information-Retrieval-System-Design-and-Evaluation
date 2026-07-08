# Information Retrieval System - Lab 5

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Lab 5 - Information Retrieval Course**  
> USTHB - Faculty of Computer Science | M2 SII  
> Dataset: MEDLINE (1,033 documents, 30 queries)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements and evaluates multiple Information Retrieval (IR) models on the MEDLINE medical literature dataset. It provides a comprehensive comparison of classical probabilistic models, vector space models, and language models, along with an interactive Streamlit interface for experimentation and visualization.

The system includes:
- **9 retrieval models** across three paradigms (probabilistic, vector-based, language modeling)
- **Comprehensive evaluation metrics** (Precision, Recall, MAP, DCG, nDCG)
- **Learning to Rank (LTR)** as a bonus feature with three approaches
- **Interactive web interface** built with Streamlit

---

## ✨ Features

- **Multiple IR Models**: BIR (classic/extended), BM25, VSM, LSI, and various Language Models
- **Full Evaluation Suite**: Precision@K, Recall, R-Precision, MAP, RR, DCG, nDCG
- **Interactive UI**: Query any model, visualize results, compare performance
- **Precision-Recall Curves**: Both interpolated and non-interpolated visualizations
- **Learning to Rank**: Pointwise, Pairwise (LambdaMART), and Listwise approaches
- **Document Preprocessing**: Tokenization, stopword removal, stemming
- **Index Generation**: Inverted index and document-term matrix construction

---

## 📁 Project Structure

### Root Files

- **`requirements.txt`**: Python dependencies (numpy, pandas, scikit-learn, streamlit, plotly, matplotlib, nltk, scipy)
- **`tree.txt`**: Complete directory tree visualization of the project structure

### 📂 `/src` - Source Code

Contains the main application entry point and core utilities:

- **`ui.py`**: Streamlit web interface for interactive IR system evaluation and comparison
- **`medline_parser.py`**: Parser for MEDLINE dataset files (MED.ALL, MED.QRY, MED.REL)
- **`preprocessing.py`**: Text preprocessing pipeline (tokenization, normalization, stemming, stopword removal)
- **`ltr_bonus.py`**: Learning to Rank implementation (Pointwise, Pairwise, Listwise approaches)

### 📂 `/models` - Retrieval Models

Implementation of all IR ranking models:

- **`Bir_classique.py`**: Classic Binary Independence Retrieval (BIR) model
- **`Bir_extended.py`**: Extended BIR with relevance feedback and query expansion
- **`BM25.py`**: Okapi BM25 probabilistic ranking function
- **`VSM.py`**: Vector Space Model with TF-IDF weighting and cosine similarity
- **`LSI.py`**: Latent Semantic Indexing using SVD dimensionality reduction
- **`ML_MLE.py`**: Maximum Likelihood Estimation language model
- **`ML_AddOne.py`**: Add-One (Laplace) smoothed language model
- **`Jelinek_Mercer.py`**: Jelinek-Mercer interpolation language model
- **`ML_Dirichlet.py`**: Dirichlet prior smoothed language model

### 📂 `/evaluation` - Metrics and Ranking

Evaluation metrics and pairwise learning utilities:

- **`metrics.py`**: Complete IR metrics implementation (Precision, Recall, MAP, RR, DCG, nDCG, R-Precision, F-measure)
- **`pairwise_gain_dcg.py`**: Pairwise gain computation using DCG objective
- **`pairwise_gain_dcg_normalized.py`**: Normalized DCG-based pairwise gain
- **`pairwise_gain_ap.py`**: Pairwise gain computation using Average Precision objective

### 📂 `/data` - Dataset Files

MEDLINE corpus and generated indices:

- **`MED.ALL`**: Complete MEDLINE document collection (1,033 medical abstracts)
- **`MED.QRY`**: Query collection (30 information needs)
- **`MED.REL`**: Relevance judgments (ground truth for evaluation)
- **`/output`**: Contains generated inverted index and document-term matrix files

### 📂 `/results` - Experimental Results

Contains evaluation outputs and visualizations:

- **Text files**: Model performance results (BIR variants, BM25, VSM, LSI, Language Models)
- **`/figures`**: Precision-Recall curves (interpolated and non-interpolated) for all models and queries
- **`/gain_dcg_normalised`**: Pairwise gain matrices and reports

### 📂 `/help` - Helper Modules

Reference implementations and utilities:

- **`BIR_BIRET_BM25.py`**: Combined BIR/BM25 reference implementation
- **`model_language.py`**: Language model utility functions and shared logic

### 📂 `/report` - Documentation

Project reports and documentation files.

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd lab5-information-retrieval
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (first run only)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 💻 Usage

### Running the Interactive UI

Launch the Streamlit web interface:

```bash
cd src
streamlit run ui.py
```

The interface provides:
- Model selection and parameter tuning
- Query input and execution
- Real-time results and metrics
- Precision-Recall curve visualization
- Document ranking comparison

### Running Individual Models

Each model can be executed standalone:

```bash
python models/BM25.py
python models/Bir_extended.py
python models/LSI.py
# ... etc
```

### Generating Indices

If you need to regenerate the inverted index and document-term matrix:

```python
from preprocessing import MEDLINEPreprocessor
from medline_parser import parse_med_all

# Parse documents
docs = parse_med_all("data/MED.ALL")

# Build indices (automatically handled by ui.py)
# Or use the "Generate index files" button in the Streamlit sidebar
```

---

## 🧮 Models Implemented

### Probabilistic Models

1. **BIR Classic**: Binary Independence Retrieval without relevance feedback
2. **BIR Extended**: Enhanced BIR with pseudo-relevance feedback and query expansion
3. **BM25**: Okapi BM25 with tunable parameters (k1, b)

### Vector Space Models

4. **VSM**: TF-IDF weighting with cosine similarity
5. **LSI**: Latent Semantic Indexing using SVD (configurable dimensionality)

### Language Models

6. **MLE**: Maximum Likelihood Estimation (no smoothing)
7. **Add-One (Laplace)**: Simple additive smoothing
8. **Jelinek-Mercer**: Linear interpolation with collection model
9. **Dirichlet**: Bayesian smoothing with Dirichlet prior

### 🎁 Bonus: Learning to Rank

10. **Pointwise**: Treats ranking as regression/classification
11. **Pairwise**: LambdaMART-style pairwise preferences
12. **Listwise**: Optimizes entire ranking list directly

---

## 📊 Evaluation Metrics

The system computes comprehensive IR metrics:

- **Precision@K** (K=5, 10): Precision at top-K results
- **Recall**: Coverage of relevant documents
- **R-Precision**: Precision at R (where R = number of relevant docs)
- **Reciprocal Rank (RR)**: Inverse rank of first relevant document
- **Average Precision (AP)**: Area under precision-recall curve
- **Mean Average Precision (MAP)**: AP averaged across queries
- **DCG@20**: Discounted Cumulative Gain at position 20
- **nDCG@20**: Normalized DCG at position 20
- **F-Measure**: Harmonic mean of precision and recall

---

## 📈 Results

The `/results` folder contains:

- **Performance reports**: Detailed metrics for each model configuration
- **Precision-Recall curves**: Visual comparison of model performance across all 30 queries
- **Pairwise gain matrices**: LTR training artifacts showing preference learning
- **Figures organized by model**: Separate subdirectories for each model variant with per-query visualizations

Key findings typically show:
- BM25 and Dirichlet LM achieve strong performance on MEDLINE
- LSI helps with synonym matching but requires tuning
- Learning to Rank can boost performance by combining model strengths

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional retrieval models (DFR, BM25+, neural models)
- More datasets beyond MEDLINE
- Query expansion techniques
- Interactive result analysis tools
- Performance optimization

Please open issues or pull requests on GitHub.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **MEDLINE Dataset**: National Library of Medicine
- **Course**: Information Retrieval, USTHB M2 SII
- **Libraries**: NumPy, pandas, scikit-learn, NLTK, Streamlit, Plotly

---

## 📧 Contact

For questions or collaboration:
- **University**: USTHB (Université des Sciences et de la Technologie Houari Boumediene)
- **Program**: Master 2 - Systèmes d'Information Intelligents
- **Course**: Information Retrieval (Recherche d'Information)

---

**Built with ❤️ for Information Retrieval research**
