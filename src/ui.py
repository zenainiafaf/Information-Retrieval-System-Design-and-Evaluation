
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="IR System Evaluation - LAB 5",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PATH SETUP
# =============================================================================
THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name.lower() == "src":
    PROJECT_ROOT = THIS_FILE.parent.parent
    SRC_DIR = THIS_FILE.parent
else:
    PROJECT_ROOT = THIS_FILE.parent
    SRC_DIR = PROJECT_ROOT / "src"

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "evaluation"
DATA_DIR = PROJECT_ROOT / "data"

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR, EVAL_DIR]:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# Plotting
# =============================================================================
PLOTLY_OK = True
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    PLOTLY_OK = False

# =============================================================================
# Import LTR Bonus Module
# =============================================================================
LTR_OK = False
try:
    from ltr_bonus import (
        create_ltr_model, 
        is_ltr_available, 
        get_available_approaches,
        get_approach_description
    )
    LTR_OK = is_ltr_available()
except Exception as e:
    LTR_OK = False
    print(f"Warning: Could not import ltr_bonus module: {e}")

# =============================================================================
# Import project modules
# =============================================================================
IMPORT_ERRORS: List[str] = []

try:
    from medline_parser import parse_med_all, parse_med_qry, parse_med_rel
except Exception as e:
    IMPORT_ERRORS.append(f"medline_parser import failed: {e!r}")

try:
    from preprocessing import MEDLINEPreprocessor
except Exception as e:
    IMPORT_ERRORS.append(f"preprocessing import failed: {e!r}")

try:
    from metrics import IRMetrics
except Exception as e:
    IMPORT_ERRORS.append(f"metrics import failed: {e!r}")

try:
    from Bir_classique import ClassicBIRModel
    from Bir_extended import ExtendedBIRModel
    from BM25 import BM25Model
    from VSM import VSMModel
    from LSI import LSIModel
    from ML_MLE import LanguageModelMLE
    from ML_AddOne import LanguageModelLaplace
    from Jelinek_Mercer import LanguageModelJelinekMercer
    from ML_Dirichlet import LanguageModelDirichlet
except Exception as e:
    IMPORT_ERRORS.append(f"models import failed: {e!r}")

if IMPORT_ERRORS:
    st.error("❌ Imports failed. Fix these first:")
    for err in IMPORT_ERRORS:
        st.code(err)
    st.stop()

# =============================================================================
# Files
# =============================================================================
MED_ALL_PATH = DATA_DIR / "MED.ALL"
MED_QRY_PATH = DATA_DIR / "MED.QRY"
MED_REL_PATH = DATA_DIR / "MED.REL"

DEFAULT_OUTPUT_DIR = DATA_DIR / "output"
DEFAULT_INVERTED = DEFAULT_OUTPUT_DIR / "inverted_index.txt"
DEFAULT_DTM = DEFAULT_OUTPUT_DIR / "document_term_matrix.txt"

# =============================================================================
# Helper functions
# =============================================================================
def find_file_in_project(filename: str, root: Path) -> Path | None:
    ignore = {".venv", "venv", "__pycache__", ".git", ".idea", ".vscode"}
    hits: List[Path] = []
    for p in root.rglob(filename):
        if any(part in ignore for part in p.parts):
            continue
        hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return hits[0]

def safe_doc_text(doc: Any) -> Tuple[str, str]:
    title = getattr(doc, "title", "") or getattr(doc, "doc_title", "") or ""
    abstract = getattr(doc, "abstract", "") or getattr(doc, "doc_abstract", "") or ""
    return str(title).strip(), str(abstract).strip()

def get_doc_id(doc: Any) -> int:
    return int(getattr(doc, "doc_id", getattr(doc, "id", -1)))

def basic_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", (text or "").lower())

def normalize_terms(preprocessor: Any, query_text: str) -> List[str]:
    if preprocessor is None:
        return basic_tokenize(query_text)
    
    for method_name in ["preprocess_text", "process_text", "preprocess_query", "tokenize"]:
        if hasattr(preprocessor, method_name):
            out = getattr(preprocessor, method_name)(query_text)
            if isinstance(out, list):
                return [str(x) for x in out if str(x).strip()]
            if isinstance(out, str):
                return basic_tokenize(out)
    return basic_tokenize(query_text)

def build_index_files(docs: List[Any], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = out_dir / "inverted_index.txt"
    dtm_path = out_dir / "document_term_matrix.txt"
    
    postings: Dict[str, Dict[int, int]] = defaultdict(dict)
    df: Counter = Counter()
    doc_lengths: Dict[int, int] = {}
    
    for doc in docs:
        doc_id = get_doc_id(doc)
        title, abstract = safe_doc_text(doc)
        tokens = basic_tokenize(f"{title} {abstract}".strip())
        tf = Counter(tokens)
        doc_lengths[doc_id] = sum(tf.values())
        
        for term, freq in tf.items():
            postings[term][doc_id] = freq
        
        for term in tf.keys():
            df[term] += 1
    
    N = len(doc_lengths) if doc_lengths else 1
    
    with inv_path.open("w", encoding="utf-8") as f_inv, dtm_path.open("w", encoding="utf-8") as f_dtm:
        for term in sorted(postings.keys()):
            idf = math.log10((N + 1) / (df[term] + 1)) + 1.0
            for doc_id in sorted(postings[term].keys()):
                freq = postings[term][doc_id]
                tfw = 1.0 + math.log10(freq) if freq > 0 else 0.0
                weight = tfw * idf
                f_inv.write(f"{term} {doc_id} {freq} {weight}\n")
                f_dtm.write(f"{doc_id} {term} {freq} {weight}\n")
    
    return inv_path, dtm_path

@st.cache_data(show_spinner=False)
def load_medline():
    docs = parse_med_all(str(MED_ALL_PATH))
    queries = parse_med_qry(str(MED_QRY_PATH))
    rel = parse_med_rel(str(MED_REL_PATH))
    return docs, queries, rel

@st.cache_resource(show_spinner=False)
def get_preprocessor():
    return MEDLINEPreprocessor()

# =============================================================================
# MAIN UI
# =============================================================================
st.title("🔬 Information Retrieval System")
st.markdown("### LAB 5")
st.markdown("---")

# Check files
missing_files = []
for p, name in [(MED_ALL_PATH, "MED.ALL"), (MED_QRY_PATH, "MED.QRY"), (MED_REL_PATH, "MED.REL")]:
    if not p.exists():
        missing_files.append(f"**{name}** → `{p}`")

if missing_files:
    st.error("Missing required files:")
    for f in missing_files:
        st.markdown(f"- {f}")
    st.stop()

# Load data
with st.spinner("Loading MEDLINE dataset..."):
    docs, queries, relevance_judgments = load_medline()
    preprocessor = get_preprocessor()

docs_by_id: Dict[int, Any] = {get_doc_id(d): d for d in docs}
query_by_id: Dict[int, Any] = {int(q.query_id): q for q in queries}

st.success(f"✅ Dataset loaded: {len(docs)} documents, {len(queries)} queries")

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Index files
    st.subheader("📁 Index Files")
    output_dir_str = st.text_input("Output folder", value=str(DEFAULT_OUTPUT_DIR))
    output_dir = Path(output_dir_str)
    
    inv_path = output_dir / "inverted_index.txt"
    dtm_path = output_dir / "document_term_matrix.txt"
    
    if not inv_path.exists():
        found = find_file_in_project("inverted_index.txt", PROJECT_ROOT)
        if found:
            inv_path = found
    if not dtm_path.exists():
        found = find_file_in_project("document_term_matrix.txt", PROJECT_ROOT)
        if found:
            dtm_path = found
    
    st.caption(f"Inverted: `{inv_path.name}`")
    st.caption(f"Doc-Term: `{dtm_path.name}`")
    
    missing = (not inv_path.exists()) or (not dtm_path.exists())
    if missing:
        st.warning("⚠️ Index files missing")
        if st.button("🛠️ Generate index files", use_container_width=True):
            with st.spinner("Generating..."):
                inv_path, dtm_path = build_index_files(docs, output_dir)
            st.success("✅ Generated!")
            st.rerun()
    else:
        st.success("✅ Index files ready")
    
    st.divider()
    
    # Model parameters
    st.subheader("🎛️ Model Parameters")
    with st.expander("LSI"):
        lsi_k = st.slider("k (dimensions)", 50, 400, 100, 10)
    
    with st.expander("BM25"):
        bm25_k1 = st.slider("k1", 0.5, 3.0, 1.2, 0.1)
        bm25_b = st.slider("b", 0.0, 1.0, 0.75, 0.05)
    
    with st.expander("Language Models"):
        jm_lambda = st.slider("Jelinek-Mercer λ", 0.05, 0.95, 0.2, 0.05)
        dir_auto = st.checkbox("Dirichlet μ auto", value=True)
        dir_mu = None if dir_auto else st.number_input("Dirichlet μ", min_value=1.0, value=1000.0, step=100.0)
    
    st.divider()
    
    # Display options
    st.subheader("📊 Display Options")
    top_k_display = st.slider("Top-K results", 5, 50, 10, 5)
    show_doc_details = st.checkbox("Show document details", value=True)
    
    st.info("ℹ️ DCG/nDCG uses fixed K=20 (as per metrics.py)")
    
    st.divider()
    
    # LTR Options
    if LTR_OK:
        st.subheader("🎁 Learning to Rank (Bonus)")
        enable_ltr = st.checkbox("Enable LTR", value=False)
        if enable_ltr:
            available_approaches = get_available_approaches()
            ltr_approach = st.selectbox(
                "LTR Approach",
                options=available_approaches,
                help="Choose the Learning to Rank approach"
            )
            
            # Show description
            approach_lower = ltr_approach.lower()
            st.info(get_approach_description(approach_lower))
            
            ltr_train_queries = st.slider("Training queries (1-10)", 1, 10, 8)
            
            # Advanced parameters (set defaults)
            n_estimators = 100
            max_depth = 10
            learning_rate = 0.1
            
            # Advanced parameters
            with st.expander("Advanced LTR Parameters"):
                if approach_lower in ["pairwise", "listwise"]:
                    n_estimators = st.slider("Number of estimators", 50, 200, 100, 10)
                if approach_lower == "pairwise":
                    max_depth = st.slider("Max depth", 5, 20, 10)
                if approach_lower == "listwise":
                    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        else:
            # Set defaults when LTR is disabled
            ltr_approach = "Pairwise"
            ltr_train_queries = 8
            n_estimators = 100
            max_depth = 10
            learning_rate = 0.1
    else:
        st.warning("⚠️ LTR not available - Install scikit-learn")
        enable_ltr = False
        ltr_approach = "Pairwise"
        ltr_train_queries = 8
        n_estimators = 100
        max_depth = 10
        learning_rate = 0.1

if missing:
    st.info("👈 Generate index files from sidebar to continue")
    st.stop()

# =============================================================================
# BUILD MODELS
# =============================================================================
@st.cache_resource(show_spinner=True)
def build_models(inv: str, dtm: str, lsi_k: int, bm25_k1: float, bm25_b: float, 
                 jm_lambda: float, dir_mu: float | None):
    bir_classic = ClassicBIRModel(inv, dtm)
    bir_extended = ExtendedBIRModel(inv, dtm)
    bm25 = BM25Model(inv, dtm, k1=bm25_k1, b=bm25_b)
    
    vsm = VSMModel()
    vsm.fit(inv, verbose=False)
    
    lsi = LSIModel(k=lsi_k)
    lsi.fit(inv, verbose=False)
    
    lm_mle = LanguageModelMLE(inv, dtm)
    lm_laplace = LanguageModelLaplace(inv, dtm)
    lm_jm = LanguageModelJelinekMercer(inv, dtm, lambda_param=jm_lambda)
    lm_dir = LanguageModelDirichlet(inv, dtm, mu_param=dir_mu)
    
    return {
        "BIR Classic (no relevance)": bir_classic,
        "BIR Classic (with relevance)": bir_classic,
        "BIR Extended (no relevance)": bir_extended,
        "BIR Extended (with relevance)": bir_extended,
        "BM25": bm25,
        "VSM (Cosine TF-IDF)": vsm,
        "LSI": lsi,
        "LM - MLE": lm_mle,
        "LM - Laplace (Add-One)": lm_laplace,
        "LM - Jelinek-Mercer": lm_jm,
        "LM - Dirichlet": lm_dir,
    }

with st.spinner("Building IR models..."):
    models = build_models(str(inv_path), str(dtm_path), lsi_k, bm25_k1, bm25_b, jm_lambda, dir_mu)

st.success(f"✅ {len(models)} models ready")

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = None
if 'ltr_model' not in st.session_state:
    st.session_state.ltr_model = None
if 'ltr_trained' not in st.session_state:
    st.session_state.ltr_trained = False

# =============================================================================
# QUERY SELECTION
# =============================================================================
st.markdown("---")
st.subheader("🔍 Query Selection")

col1, col2 = st.columns([1, 2])

with col1:
    available_queries = sorted([qid for qid in query_by_id.keys() if qid <= 10])
    qid = st.selectbox("Select Query (I1-I10)", options=available_queries, format_func=lambda x: f"Query {x}")

with col2:
    q_obj = query_by_id[int(qid)]
    q_text = getattr(q_obj, "text", "") or ""
    relevant_docs = relevance_judgments.get(int(qid), [])
    rel_set = set(relevant_docs)
    
    st.markdown(f"**Query Text:**")
    st.info(q_text)
    st.caption(f"**Relevant documents:** {len(relevant_docs)} → {relevant_docs[:10]}{'...' if len(relevant_docs) > 10 else ''}")

# =============================================================================
# MODEL SELECTION WITH FIXED BUTTONS
# =============================================================================
st.markdown("---")
st.subheader("🎯 Model Selection")

# Group models by category
model_categories = {
    "Probabilistic Models": [
        "BIR Classic (no relevance)",
        "BIR Classic (with relevance)",
        "BIR Extended (no relevance)",
        "BIR Extended (with relevance)",
        "BM25"
    ],
    "Vector Space Models": [
        "VSM (Cosine TF-IDF)",
        "LSI"
    ],
    "Language Models": [
        "LM - MLE",
        "LM - Laplace (Add-One)",
        "LM - Jelinek-Mercer",
        "LM - Dirichlet"
    ]
}

# Quick select buttons
st.markdown("**Quick Selection:**")
qcol1, qcol2, qcol3, qcol4 = st.columns(4)

with qcol1:
    if st.button("✅ Select All", use_container_width=True):
        st.session_state.selected_models = list(models.keys())
        st.rerun()

with qcol2:
    if st.button("⭐ Best Performers", use_container_width=True):
        st.session_state.selected_models = ["BM25", "VSM (Cosine TF-IDF)", "LSI", "LM - Dirichlet"]
        st.rerun()

with qcol3:
    if st.button("🎲 Probabilistic", use_container_width=True):
        st.session_state.selected_models = model_categories["Probabilistic Models"]
        st.rerun()

with qcol4:
    if st.button("❌ Clear All", use_container_width=True):
        st.session_state.selected_models = []
        st.rerun()

st.markdown("---")

# Model checkboxes
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Probabilistic Models**")
    for model in model_categories["Probabilistic Models"]:
        checked = model in st.session_state.selected_models
        if st.checkbox(model, value=checked, key=f"cb_{model}"):
            if model not in st.session_state.selected_models:
                st.session_state.selected_models.append(model)
        else:
            if model in st.session_state.selected_models:
                st.session_state.selected_models.remove(model)

with col2:
    st.markdown("**Vector Space Models**")
    for model in model_categories["Vector Space Models"]:
        checked = model in st.session_state.selected_models
        if st.checkbox(model, value=checked, key=f"cb_{model}"):
            if model not in st.session_state.selected_models:
                st.session_state.selected_models.append(model)
        else:
            if model in st.session_state.selected_models:
                st.session_state.selected_models.remove(model)

with col3:
    st.markdown("**Language Models**")
    for model in model_categories["Language Models"]:
        checked = model in st.session_state.selected_models
        if st.checkbox(model, value=checked, key=f"cb_{model}"):
            if model not in st.session_state.selected_models:
                st.session_state.selected_models.append(model)
        else:
            if model in st.session_state.selected_models:
                st.session_state.selected_models.remove(model)

selected_models = st.session_state.selected_models

# =============================================================================
# LTR TRAINING
# =============================================================================

if enable_ltr and LTR_OK:
    st.markdown("---")
    st.subheader("🎓 Train Learning to Rank Model")
    
    # Optional: Show what is currently trained
    if st.session_state.ltr_trained:
        current_approach = st.session_state.get('ltr_approach', 'Unknown')
        st.info(f"✅ Current active model: **{current_approach}**. You can retrain below.")

    st.info(f"📚 Approach: **{ltr_approach}** | Training on queries 1-{ltr_train_queries}")
    
    if st.button("🚀 Train LTR Model", type="primary", use_container_width=True):
        # ... (keep the rest of the logic exactly the same)
        with st.spinner(f"Training {ltr_approach} LTR model..."):
            # Create LTR model based on selected approach
            approach_lower = ltr_approach.lower()
            kwargs = {}
            
            if approach_lower in ["pairwise", "listwise"]:
                kwargs['n_estimators'] = n_estimators
            if approach_lower == "pairwise":
                kwargs['max_depth'] = max_depth
            if approach_lower == "listwise":
                kwargs['learning_rate'] = learning_rate
            
            ltr = create_ltr_model(approach_lower, **kwargs)
            
            if ltr is None:
                st.error("❌ Could not create LTR model")
            else:
                # Collect training data
                training_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                train_query_ids = [i for i in range(1, ltr_train_queries + 1)]
                
                for idx, train_qid in enumerate(train_query_ids):
                    status_text.text(f"Collecting features for Query {train_qid}...")
                    progress_bar.progress(idx / len(train_query_ids))
                    
                    q_obj_train = query_by_id[train_qid]
                    q_text_train = getattr(q_obj_train, "text", "")
                    query_terms_train = normalize_terms(preprocessor, q_text_train)
                    relevant_docs_train = relevance_judgments.get(train_qid, [])
                    
                    # Get scores from all models
                    model_scores = {}
                    for model_name, model in models.items():
                        if model_name.endswith("(with relevance)"):
                            ranked = model.rank_documents(query_terms_train, relevant_docs=relevant_docs_train, top_k=None)
                        else:
                            ranked = model.rank_documents(query_terms_train, top_k=None)
                        
                        model_scores[model_name] = {int(d): float(s) for d, s in ranked}
                    
                    training_data.append({
                        'model_scores': model_scores,
                        'relevant_docs': relevant_docs_train,
                        'all_docs': list(docs_by_id.keys())
                    })
                
                # Train LTR
                status_text.text(f"Training {ltr_approach} model...")
                success = ltr.train(training_data, lambda p: progress_bar.progress(p))
                
                progress_bar.empty()
                status_text.empty()
                
                if success:
                    st.session_state.ltr_model = ltr
                    st.session_state.ltr_approach = ltr_approach
                    st.session_state.ltr_trained = True
                    st.success(f"✅ {ltr_approach} LTR Model trained successfully!")
                    
                    # Show feature importance (if available)
                    importance = ltr.get_feature_importance()
                    if importance:
                        st.markdown("**Feature Importance (Model Contributions):**")
                        imp_df = pd.DataFrame(list(importance.items()), columns=['Model', 'Importance'])
                        imp_df = imp_df.sort_values('Importance', ascending=False)
                        st.dataframe(imp_df, use_container_width=True)
                else:
                    st.error("❌ LTR training failed")

# =============================================================================
# RUN EVALUATION
# =============================================================================
st.markdown("---")

# Check if we need to rerun
need_rerun = st.session_state.current_query != qid

run = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

if run or (need_rerun and st.session_state.evaluation_results is not None):
    if not selected_models:
        st.warning("⚠️ Please select at least one model")
        st.session_state.evaluation_results = None
    else:
        query_terms = normalize_terms(preprocessor, q_text)
        if not query_terms:
            st.error("❌ Query preprocessing produced no terms")
            st.session_state.evaluation_results = None
        else:
            st.session_state.current_query = qid
            
            st.markdown("### 📊 Evaluation Results")
            
            metrics_engine = IRMetrics(relevance_judgments, model_name="Streamlit_UI")
            relevance_scores_binary = {int(d): 1.0 for d in relevant_docs}
            
            rows = []
            pr_curves: Dict[str, List[Dict[str, float]]] = {}
            pr_curves_interp: Dict[str, List[Tuple[float, float]]] = {}
            all_rankings: Dict[str, List[Tuple[int, float]]] = {}
            
                       # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Collect model scores for LTR - MUST USE ALL MODELS (not just selected ones)
            model_scores_for_ltr = {}
            
            # IMPORTANT: Always collect scores from ALL models for LTR feature consistency
            for model_name, model in models.items():
                # Rank documents with this model
                if model_name.endswith("(with relevance)"):
                    ranked = model.rank_documents(query_terms, relevant_docs=relevant_docs, top_k=None)
                else:
                    ranked = model.rank_documents(query_terms, top_k=None)
                
                # Store scores for LTR (needed for feature extraction)
                model_scores_for_ltr[model_name] = {int(d): float(s) for d, s in ranked}
            
            # Now evaluate only the SELECTED models for display
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Evaluating {model_name}...")
                progress_bar.progress((idx + 1) / len(selected_models))
                
                # Use the already computed rankings
                ranked = [(d, s) for d, s in model_scores_for_ltr[model_name].items()]
                ranked.sort(key=lambda x: x[1], reverse=True)
                
                all_rankings[model_name] = ranked
                ranked_list = [int(d) for d, _ in ranked]

                model_scores_for_ltr[model_name] = {int(d): float(s) for d, s in ranked}
                
                # Evaluate
                res = metrics_engine.evaluate_query(
                    query_id=int(qid),
                    ranked_list=ranked_list,
                    relevance_scores=relevance_scores_binary,
                    verbose=False
                )
                
                pr_curves[model_name] = res.get("precision_recall_curve", [])
                pr_curves_interp[model_name] = res.get("interpolated_curve", [])
                
                rows.append({
                    "Model": model_name,
                    "P@5": res.get("p_at_5", 0.0),
                    "P@10": res.get("p_at_10", 0.0),
                    "R-Precision": res.get("r_precision", 0.0),
                    "RR": res.get("reciprocal_rank", 0.0),
                    "AP": res.get("average_precision", 0.0),
                    "AP (interp)": res.get("average_precision_interpolated", 0.0),
                    "DCG@20": res.get("dcg_at_20", 0.0),
                    "nDCG@20": res.get("ndcg_at_20", 0.0),
                })
            
            # LTR Evaluation
            if enable_ltr and st.session_state.ltr_trained:
                status_text.text("Evaluating LTR model...")
                
                ltr = st.session_state.ltr_model
                ltr_approach_name = st.session_state.get('ltr_approach', 'Unknown')
                ltr_scores = ltr.predict_scores(model_scores_for_ltr, list(docs_by_id.keys()))
                ltr_ranked = sorted(ltr_scores.items(), key=lambda x: x[1], reverse=True)
                ltr_ranked_list = [int(d) for d, _ in ltr_ranked]
                
                ltr_display_name = f"🎁 LTR ({ltr_approach_name})"
                all_rankings[ltr_display_name] = ltr_ranked
                
                res_ltr = metrics_engine.evaluate_query(
                    query_id=int(qid),
                    ranked_list=ltr_ranked_list,
                    relevance_scores=relevance_scores_binary,
                    verbose=False
                )
                
                pr_curves[ltr_display_name] = res_ltr.get("precision_recall_curve", [])
                pr_curves_interp[ltr_display_name] = res_ltr.get("interpolated_curve", [])
                
                rows.append({
                    "Model": ltr_display_name,
                    "P@5": res_ltr.get("p_at_5", 0.0),
                    "P@10": res_ltr.get("p_at_10", 0.0),
                    "R-Precision": res_ltr.get("r_precision", 0.0),
                    "RR": res_ltr.get("reciprocal_rank", 0.0),
                    "AP": res_ltr.get("average_precision", 0.0),
                    "AP (interp)": res_ltr.get("average_precision_interpolated", 0.0),
                    "DCG@20": res_ltr.get("dcg_at_20", 0.0),
                    "nDCG@20": res_ltr.get("ndcg_at_20", 0.0),
                })
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state.evaluation_results = {
                'df': pd.DataFrame(rows).sort_values(by="AP", ascending=False),
                'pr_curves': pr_curves,
                'pr_curves_interp': pr_curves_interp,
                'all_rankings': all_rankings,
                'qid': qid,
                'q_text': q_text,
                'relevant_docs': relevant_docs,
                'rel_set': rel_set
            }

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
if st.session_state.evaluation_results is not None:
    results = st.session_state.evaluation_results
    df = results['df']
    pr_curves = results['pr_curves']
    pr_curves_interp = results['pr_curves_interp']
    all_rankings = results['all_rankings']
    rel_set = results['rel_set']
    
    # Metrics table
    st.markdown("#### 📈 Metrics Summary")
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    styled_df = df.style.apply(highlight_max, subset=['P@5', 'P@10', 'R-Precision', 'RR', 'AP', 'nDCG@20'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"query_{qid}_results.csv",
        mime="text/csv"
    )
    
    # Top-K results
    st.markdown("---")
    st.markdown(f"#### 🎯 Top-{top_k_display} Retrieved Documents")
    
    for model_name in all_rankings.keys():
        with st.expander(f"**{model_name}** - Top {top_k_display} Results", expanded=False):
            ranked = all_rankings[model_name][:top_k_display]
            
            for rank, (doc_id, score) in enumerate(ranked, 1):
                doc = docs_by_id.get(int(doc_id))
                title, abstract = safe_doc_text(doc) if doc else ("", "")
                is_rel = int(doc_id) in rel_set
                
                col_rank, col_rel, col_info = st.columns([0.5, 0.5, 9])
                
                with col_rank:
                    st.markdown(f"**#{rank}**")
                
                with col_rel:
                    st.markdown("✅" if is_rel else "❌")
                
                with col_info:
                    st.markdown(f"**Doc {doc_id}** | Score: `{score:.6f}`")
                    
                    if show_doc_details:
                        if title:
                            st.markdown(f"**Title:** *{title}*")
                        if abstract:
                            abstract_preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                            st.markdown(f"**Abstract:** {abstract_preview}")
                
                st.divider()
    
    # Charts
    if PLOTLY_OK:
        st.markdown("---")
        st.markdown("#### 📉 Precision-Recall Curves")
        
        tab1, tab2 = st.tabs(["Non-Interpolated", "Interpolated"])
        
        with tab1:
            st.markdown("**Standard Precision-Recall Curve**")
            fig1 = go.Figure()
            
            for name, points in pr_curves.items():
                rel_points = [p for p in points if p.get("is_relevant")]
                if not rel_points:
                    continue
                recalls = [p["recall"] for p in rel_points]
                precisions = [p["precision"] for p in rel_points]
                fig1.add_trace(go.Scatter(
                    x=recalls, y=precisions, mode="lines+markers", name=name,
                    hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                ))
            
            fig1.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1.05]),
                height=600, hovermode='closest',
                legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            st.markdown("**Interpolated Precision-Recall Curve**")
            fig2 = go.Figure()
            
            for name, interp_curve in pr_curves_interp.items():
                if not interp_curve:
                    continue
                recalls = [p[0] for p in interp_curve]
                precisions = [p[1] for p in interp_curve]
                fig2.add_trace(go.Scatter(
                    x=recalls, y=precisions, mode="lines+markers", name=name,
                    hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                ))
            
            fig2.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1.05]),
                height=600, hovermode='closest',
                legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Comparison
        st.markdown("---")
        st.markdown("#### 📊 Metrics Comparison")
        
        metric_choice = st.selectbox(
            "Select metric to compare",
            options=["AP", "P@5", "P@10", "R-Precision", "RR", "DCG@20", "nDCG@20"],
            key="metric_selector"
        )
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df["Model"], y=df[metric_choice],
            text=df[metric_choice].round(4), textposition='auto',
        ))
        
        fig3.update_layout(
            title=f"{metric_choice} Comparison",
            xaxis_title="Model", yaxis_title=metric_choice,
            height=500, showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# PAIRWISE COMPARISON SECTION (Ajouter après la section DISPLAY RESULTS)
# =============================================================================

st.markdown("---")
st.markdown("### ⚖️ Pairwise Model Comparison")

# Import pairwise gain modules
try:
    import sys
    pairwise_dir = PROJECT_ROOT / "evaluation"
    if str(pairwise_dir) not in sys.path:
        sys.path.insert(0, str(pairwise_dir))
    
    from pairwise_gain_ap import PairwiseGainCalculator as PairwiseAP, load_all_results as load_ap
    from pairwise_gain_dcg import PairwiseGainCalculator as PairwiseDCG, load_all_results as load_dcg
    from pairwise_gain_dcg_normalized import PairwiseGainCalculator as PairwiseNDCG, load_all_results as load_ndcg
    
    PAIRWISE_OK = True
except Exception as e:
    st.warning(f"⚠️ Pairwise comparison not available: {e}")
    PAIRWISE_OK = False

if PAIRWISE_OK:
    st.info("📊 Compare two models using Average Precision (AP), DCG@20, or nDCG@20")
    
    # Load available results
    results_dir = PROJECT_ROOT / "results"
    
    if results_dir.exists():
        # Metric selection
        comparison_metric = st.selectbox(
            "Select comparison metric",
            options=["Average Precision (AP)", "DCG@20", "nDCG@20"],
            key="comparison_metric"
        )
        
        # Load data based on metric
        with st.spinner(f"Loading {comparison_metric} data..."):
            query_ids = list(range(1, 11))  # I1-I10
            
            if comparison_metric == "Average Precision (AP)":
                metric_data = load_ap(str(results_dir), query_ids)
                calculator = PairwiseAP(metric_data) if metric_data else None
                metric_key = "AP"
            elif comparison_metric == "DCG@20":
                metric_data = load_dcg(str(results_dir), query_ids)
                calculator = PairwiseDCG(metric_data) if metric_data else None
                metric_key = "DCG@20"
            else:  # nDCG@20
                metric_data = load_ndcg(str(results_dir), query_ids)
                calculator = PairwiseNDCG(metric_data) if metric_data else None
                metric_key = "nDCG@20"
        
        if calculator and metric_data:
            available_models = sorted(metric_data.keys())
            
            st.success(f"✅ Loaded {len(available_models)} models for comparison")
            
            # Model selection
            col1, col2 = st.columns(2)
            
            with col1:
                model_a = st.selectbox(
                    "Select Model A",
                    options=available_models,
                    key="model_a_selector"
                )
            
            with col2:
                model_b = st.selectbox(
                    "Select Model B",
                    options=[m for m in available_models if m != model_a],
                    key="model_b_selector"
                )
            
            if st.button("🔄 Compare Models", type="primary", use_container_width=True):
                st.markdown("---")
                
                # Generate comparison
                comparison_df = calculator.generate_comparison_table(model_a, model_b, query_ids)
                
                # Display comparison table
                st.markdown(f"#### 📊 Comparison: **{model_a}** vs **{model_b}**")
                st.markdown(f"**Metric:** {comparison_metric}")
                
                # Style the dataframe
                def highlight_winner(row):
                    if row['Query'] == 'Mean':
                        return ['background-color: #e6f3ff'] * len(row)
                    
                    gain = row[f'Gain {model_a} vs {model_b} (%)']
                    
                    if gain >= 5:
                        # Model A is better
                        return ['', 'background-color: #90EE90', '', 'background-color: #90EE90']
                    elif gain < 5:
                        # Model B is better
                        return ['', '', 'background-color: #FFB6C1', 'background-color: #FFB6C1']
                    else:
                        # Tie
                        return ['', '', '', '']
                
                styled_comparison = comparison_df.style.apply(highlight_winner, axis=1)
                st.dataframe(styled_comparison, use_container_width=True, height=450)
                
                # Analysis
                mean_gain = calculator.calculate_mean_gain(model_a, model_b, query_ids)
                
                st.markdown("---")
                st.markdown("#### 📈 Analysis")
                
                col_a1, col_a2, col_a3 = st.columns(3)
                
                with col_a1:
                    if mean_gain >= 5:
                        st.metric(
                            label="Winner",
                            value=model_a,
                            delta=f"+{mean_gain:.2f}%",
                            delta_color="normal"
                        )
                    elif mean_gain < 5:
                        st.metric(
                            label="Winner",
                            value=model_b,
                            delta=f"{abs(mean_gain):.2f}%",
                            delta_color="inverse"
                        )
                    else:
                        st.metric(
                            label="Result",
                            value="Tie",
                            delta="0.00%"
                        )
                
                with col_a2:
                    wins_a = sum(1 for qid in query_ids 
                               if calculator.calculate_gain(model_a, model_b, qid) >= 5)
                    st.metric(f"{model_a} Wins", wins_a)
                
                with col_a3:
                    wins_b = sum(1 for qid in query_ids 
                               if calculator.calculate_gain(model_a, model_b, qid) < 5)
                    st.metric(f"{model_b} Wins", wins_b)
                
                # Visualization
                if PLOTLY_OK:
                    st.markdown("---")
                    st.markdown("#### 📉 Visual Comparison")
                    
                    # Bar chart comparison
                    fig_comp = go.Figure()
                    
                    # Remove 'Mean' row for visualization
                    comp_viz = comparison_df[comparison_df['Query'] != 'Mean'].copy()
                    
                    fig_comp.add_trace(go.Bar(
                        name=model_a,
                        x=comp_viz['Query'],
                        y=comp_viz[f'{model_a} {metric_key}'],
                        marker_color='lightblue'
                    ))
                    
                    fig_comp.add_trace(go.Bar(
                        name=model_b,
                        x=comp_viz['Query'],
                        y=comp_viz[f'{model_b} {metric_key}'],
                        marker_color='lightcoral'
                    ))
                    
                    fig_comp.update_layout(
                        title=f"{metric_key} Comparison per Query",
                        xaxis_title="Query",
                        yaxis_title=metric_key,
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Gain chart
                    fig_gain = go.Figure()
                    
                    gains = [calculator.calculate_gain(model_a, model_b, qid) 
                            for qid in query_ids]
                    colors = ['green' if g > 0 else 'red' if g < 0 else 'gray' 
                             for g in gains]
                    
                    fig_gain.add_trace(go.Bar(
                        x=[f'Q{qid}' for qid in query_ids],
                        y=gains,
                        marker_color=colors,
                        text=[f'{g:+.2f}%' for g in gains],
                        textposition='outside'
                    ))
                    
                    fig_gain.update_layout(
                        title=f"Gain % ({model_a} vs {model_b})",
                        xaxis_title="Query",
                        yaxis_title="Gain (%)",
                        height=400,
                        showlegend=False
                    )
                    
                    fig_gain.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    st.plotly_chart(fig_gain, use_container_width=True)
                
                # Download comparison
                csv_comparison = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison (CSV)",
                    data=csv_comparison,
                    file_name=f"{model_a}_vs_{model_b}_{metric_key}.csv",
                    mime="text/csv"
                )
                
                # Detailed interpretation
                with st.expander("📚 How to interpret the results"):
                    st.markdown(f"""
                    **Gain % Calculation:**
                    - Gain % = ((Model A {metric_key} - Model B {metric_key}) / Model B {metric_key}) × 100
                    
                    **Interpretation:**
                    - **gain (=>5%)**: Model A performs better than Model B
                    - **gain (<5%)**: Model B performs better than Model A
                    
                    **Color coding:**
                    - 🟢 Green: Model A wins on this query
                    - 🔴 Red: Model B wins on this query
                    - ⚪ Gray: Tie
                    
                    **Mean row:**
                    - Shows average {metric_key} and average gain across all queries
                    - The model with higher mean {metric_key} is generally better
                    """)
        else:
            st.warning("⚠️ No comparison data available. Make sure you have run evaluations and generated result files.")
    else:
        st.warning(f"⚠️ Results directory not found: {results_dir}")
else:
    st.info("ℹ️ Pairwise comparison requires pairwise_gain_*.py modules in evaluation/ folder")

if st.session_state.evaluation_results is None:
    st.info("👆 Select models and click 'Run Evaluation' to see results")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>IR System Evaluation - LAB 5</strong></p>
    <p>USTHB - Faculty of Computer Science | M2 SII - Information Retrieval</p>
    <p>Dataset: MEDLINE (1,033 documents, 30 queries)</p>
    <p>🎁 Bonus: Learning to Rank (Pointwise, Pairwise, Listwise approaches)</p>
</div>
""", unsafe_allow_html=True)