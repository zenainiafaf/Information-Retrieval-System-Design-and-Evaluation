# ltr_bonus.py - Learning to Rank Implementation for LAB 5 Bonus
"""
Learning to Rank (LTR) Module - LAB 5 Bonus

This module implements three LTR approaches:
1. Pairwise (RankNet-style with Random Forest)
2. Pointwise (Classification/Regression)
3. Listwise (ListNet approximation)

Main Features:
- Multiple approaches (Pointwise, Pairwise, Listwise)
- Feature extraction from base IR models
- Training on query-document pairs
- Prediction for new queries
- Feature importance analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

# Check if scikit-learn is available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# BASE LTR CLASS
# =============================================================================
class BaseLTR:
    """Base class for Learning to Rank models"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_names = []
    
    def extract_features(self, doc_scores: Dict[str, float], doc_id: int) -> np.ndarray:
        """
        Extract features from multiple model scores
        
        Args:
            doc_scores: Dictionary of {model_name: {doc_id: score}}
            doc_id: Document ID to extract features for
            
        Returns:
            Feature vector (numpy array)
        """
        features = []
        self.feature_names = sorted(doc_scores.keys())
        
        for model_name in self.feature_names:
            features.append(doc_scores[model_name].get(doc_id, 0.0))
        
        return np.array(features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for each base model
        
        Returns:
            Dictionary of {model_name: importance_score}
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))


# =============================================================================
# POINTWISE LTR - CLASSIFICATION APPROACH
# =============================================================================
class PointwiseLTR(BaseLTR):
    """
    Pointwise Learning to Rank using Classification
    
    Treats ranking as a binary classification problem:
    - Class 1: Relevant documents
    - Class 0: Non-relevant documents
    
    Uses Logistic Regression for simplicity and interpretability.
    """
    
    def __init__(self):
        super().__init__()
        if SKLEARN_AVAILABLE:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def create_training_data(self, all_model_scores: Dict[str, Dict[int, float]], 
                            relevant_docs: List[int], 
                            all_docs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pointwise training data
        
        Args:
            all_model_scores: Scores from all models
            relevant_docs: List of relevant document IDs
            all_docs: List of all document IDs
            
        Returns:
            X (features), y (labels)
        """
        X = []
        y = []
        
        relevant_set = set(relevant_docs)
        
        for doc_id in all_docs:
            features = self.extract_features(all_model_scores, doc_id)
            X.append(features)
            y.append(1 if doc_id in relevant_set else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, queries_data: List[Dict], progress_callback: Optional[Callable] = None) -> bool:
        """
        Train pointwise LTR model
        
        Args:
            queries_data: List of query data dictionaries
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if training successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            return False
        
        all_X = []
        all_y = []
        
        for idx, qdata in enumerate(queries_data):
            if progress_callback:
                progress_callback(idx / len(queries_data))
            
            X, y = self.create_training_data(
                qdata['model_scores'],
                qdata['relevant_docs'],
                qdata['all_docs']
            )
            
            all_X.append(X)
            all_y.append(y)
        
        if not all_X:
            return False
        
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_scores(self, all_model_scores: Dict[str, Dict[int, float]], 
                      all_docs: List[int]) -> Dict[int, float]:
        """
        Predict relevance scores for documents
        
        Args:
            all_model_scores: Scores from all models
            all_docs: List of document IDs
            
        Returns:
            Dictionary of {doc_id: predicted_score}
        """
        if not self.is_trained:
            return {doc_id: 0.0 for doc_id in all_docs}
        
        scores = {}
        for doc_id in all_docs:
            features = self.extract_features(all_model_scores, doc_id)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            # Use probability of being relevant
            score = self.model.predict_proba(features_scaled)[0][1]
            scores[doc_id] = score
        
        return scores


# =============================================================================
# PAIRWISE LTR - RANKNET APPROACH
# =============================================================================
class PairwiseLTR(BaseLTR):
    """
    Pairwise Learning to Rank using RankNet approach
    
    Learns from pairs of documents:
    - Relevant doc should rank higher than non-relevant doc
    
    Uses Random Forest Classifier for robustness.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__()
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
    
    def create_pairwise_data(self, all_model_scores: Dict[str, Dict[int, float]], 
                            relevant_docs: List[int], 
                            all_docs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pairwise training data
        
        Creates pairs: 
        1. (relevant, non_relevant) -> Label 1
        2. (non_relevant, relevant) -> Label 0 (Symmetric pair for balance)
        
        Args:
            all_model_scores: Scores from all models
            relevant_docs: List of relevant document IDs
            all_docs: List of all document IDs
            
        Returns:
            X (feature differences), y (labels)
        """
        X_pairs = []
        y_pairs = []
        
        relevant_set = set(relevant_docs)
        
        # Create pairs: relevant vs non-relevant
        for rel_doc in relevant_docs:
            for non_rel_doc in all_docs:
                if non_rel_doc not in relevant_set:
                    # Features for relevant doc
                    feat_rel = self.extract_features(all_model_scores, rel_doc)
                    # Features for non-relevant doc
                    feat_non_rel = self.extract_features(all_model_scores, non_rel_doc)
                    
                    # Positive Pair: (Rel - NonRel) -> 1
                    X_pairs.append(feat_rel - feat_non_rel)
                    y_pairs.append(1)

                    # Negative Pair: (NonRel - Rel) -> 0
                    # This ensures the classifier sees both classes and learns directionality
                    X_pairs.append(feat_non_rel - feat_rel)
                    y_pairs.append(0)
        
        return np.array(X_pairs), np.array(y_pairs)
    
    def train(self, queries_data: List[Dict], progress_callback: Optional[Callable] = None) -> bool:
        """
        Train pairwise LTR model
        
        Args:
            queries_data: List of query data dictionaries
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if training successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            return False
        
        all_X = []
        all_y = []
        
        for idx, qdata in enumerate(queries_data):
            if progress_callback:
                progress_callback(idx / len(queries_data))
            
            X_pairs, y_pairs = self.create_pairwise_data(
                qdata['model_scores'],
                qdata['relevant_docs'],
                qdata['all_docs']
            )
            
            if len(X_pairs) > 0:
                all_X.append(X_pairs)
                all_y.append(y_pairs)
        
        if not all_X:
            return False
        
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_scores(self, all_model_scores: Dict[str, Dict[int, float]], 
                      all_docs: List[int]) -> Dict[int, float]:
        """
        Predict relevance scores for documents
        
        Uses pairwise comparison probabilities as scores.
        
        Args:
            all_model_scores: Scores from all models
            all_docs: List of document IDs
            
        Returns:
            Dictionary of {doc_id: predicted_score}
        """
        if not self.is_trained:
            return {doc_id: 0.0 for doc_id in all_docs}
        
        scores = {}
        for doc_id in all_docs:
            features = self.extract_features(all_model_scores, doc_id)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            # Use probability of ranking higher
            score = self.model.predict_proba(features_scaled)[0][1]
            scores[doc_id] = score
        
        return scores


# =============================================================================
# LISTWISE LTR - GRADIENT BOOSTING APPROACH
# =============================================================================
class ListwiseLTR(BaseLTR):
    """
    Listwise Learning to Rank using Gradient Boosting
    
    Learns to optimize list-level metrics directly.
    Uses gradient boosting with ranking objective.
    
    Note: This is a simplified version. True ListNet requires
    specialized loss functions.
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        super().__init__()
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=5,
                random_state=42
            )
    
    def create_listwise_data(self, all_model_scores: Dict[str, Dict[int, float]], 
                            relevant_docs: List[int], 
                            all_docs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create listwise training data
        
        Assigns scores based on relevance:
        - Relevant docs: score = 1.0
        - Non-relevant docs: score = 0.0
        
        Args:
            all_model_scores: Scores from all models
            relevant_docs: List of relevant document IDs
            all_docs: List of all document IDs
            
        Returns:
            X (features), y (relevance scores)
        """
        X = []
        y = []
        
        relevant_set = set(relevant_docs)
        
        for doc_id in all_docs:
            features = self.extract_features(all_model_scores, doc_id)
            X.append(features)
            y.append(1.0 if doc_id in relevant_set else 0.0)
        
        return np.array(X), np.array(y)
    
    def train(self, queries_data: List[Dict], progress_callback: Optional[Callable] = None) -> bool:
        """
        Train listwise LTR model
        
        Args:
            queries_data: List of query data dictionaries
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if training successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            return False
        
        all_X = []
        all_y = []
        
        for idx, qdata in enumerate(queries_data):
            if progress_callback:
                progress_callback(idx / len(queries_data))
            
            X, y = self.create_listwise_data(
                qdata['model_scores'],
                qdata['relevant_docs'],
                qdata['all_docs']
            )
            
            all_X.append(X)
            all_y.append(y)
        
        if not all_X:
            return False
        
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Gradient Boosting
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_scores(self, all_model_scores: Dict[str, Dict[int, float]], 
                      all_docs: List[int]) -> Dict[int, float]:
        """
        Predict relevance scores for documents
        
        Args:
            all_model_scores: Scores from all models
            all_docs: List of document IDs
            
        Returns:
            Dictionary of {doc_id: predicted_score}
        """
        if not self.is_trained:
            return {doc_id: 0.0 for doc_id in all_docs}
        
        scores = {}
        for doc_id in all_docs:
            features = self.extract_features(all_model_scores, doc_id)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            score = self.model.predict(features_scaled)[0]
            scores[doc_id] = max(0.0, score)  # Ensure non-negative
        
        return scores


# =============================================================================
# LTR FACTORY
# =============================================================================
def create_ltr_model(approach: str = "pairwise", **kwargs) -> Optional[BaseLTR]:
    """
    Factory function to create LTR models
    
    Args:
        approach: One of "pointwise", "pairwise", "listwise"
        **kwargs: Additional arguments for specific models
        
    Returns:
        LTR model instance or None if approach not recognized
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    approach = approach.lower()
    
    if approach == "pointwise":
        return PointwiseLTR()
    elif approach == "pairwise":
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        return PairwiseLTR(n_estimators=n_estimators, max_depth=max_depth)
    elif approach == "listwise":
        n_estimators = kwargs.get('n_estimators', 100)
        learning_rate = kwargs.get('learning_rate', 0.1)
        return ListwiseLTR(n_estimators=n_estimators, learning_rate=learning_rate)
    else:
        return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def is_ltr_available() -> bool:
    """Check if LTR functionality is available (scikit-learn installed)"""
    return SKLEARN_AVAILABLE


def get_available_approaches() -> List[str]:
    """Get list of available LTR approaches"""
    if SKLEARN_AVAILABLE:
        return ["Pointwise", "Pairwise", "Listwise"]
    return []


def get_approach_description(approach: str) -> str:
    """Get description for each LTR approach"""
    descriptions = {
        "pointwise": "Classification: Treats each document independently. Uses Logistic Regression.",
        "pairwise": "RankNet: Learns from document pairs. Uses Random Forest for pairwise preferences.",
        "listwise": "List-based: Optimizes for entire result list. Uses Gradient Boosting."
    }
    return descriptions.get(approach.lower(), "Unknown approach")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("LEARNING TO RANK - LAB 5 BONUS MODULE")
    print("=" * 80)
    
    if not SKLEARN_AVAILABLE:
        print("\n❌ scikit-learn not installed!")
        print("Install it with: pip install scikit-learn")
    else:
        print("\n✅ scikit-learn available")
        print(f"\nAvailable approaches: {', '.join(get_available_approaches())}")
        
        for approach in ["pointwise", "pairwise", "listwise"]:
            print(f"\n{approach.upper()}:")
            print(f"  {get_approach_description(approach)}")
        
        print("\n" + "=" * 80)
        print("Example: Creating a Pairwise LTR model")
        print("=" * 80)
        
        ltr = create_ltr_model("pairwise")
        if ltr:
            print(f"✅ Created: {ltr.__class__.__name__}")
            print(f"   Trained: {ltr.is_trained}")
        
        print("\n" + "=" * 80)