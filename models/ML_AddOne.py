import math
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple


class LanguageModelLaplace:
    """Language Model avec Add-One Smoothing (Laplace)"""
    
    def __init__(self, inverted_index_path, doc_term_matrix_path):
        """
        Args:
            inverted_index_path: Chemin vers l'inverted index
            doc_term_matrix_path: Chemin vers la matrice document-terme
        """
        # Charger l'index inversé (fréquences des termes)
        self.inverted_index = self._load_inverted_index(inverted_index_path)
        
        # Charger les longueurs des documents
        self.doc_lengths = self._load_doc_lengths(doc_term_matrix_path)
        
        # Nombre de documents
        self.N = len(self.doc_lengths)
        
        # Liste des doc_ids
        self.doc_ids = sorted(list(self.doc_lengths.keys()))
        
        # Taille du vocabulaire (nombre de termes uniques)
        self.V = len(self.inverted_index)
    
    def _load_inverted_index(self, filepath):
        """Charge l'inverted index depuis le fichier"""
        inverted_index = defaultdict(dict)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    term = parts[0]
                    doc_id = int(parts[1])
                    freq = int(parts[2])  # Fréquence brute
                    inverted_index[term][doc_id] = freq
        return inverted_index
    
    def _load_doc_lengths(self, filepath):
        """Charge les longueurs des documents (|d| = somme des fréquences)"""
        doc_lengths = defaultdict(int)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    doc_id = int(parts[0])
                    freq = int(parts[2])
                    doc_lengths[doc_id] += freq
        return doc_lengths
    
    def process_query(self, query_text):
        """Prétraite une requête (simple split en minuscules)"""
        return query_text.lower().split()
    
    def get_term_freq(self, term, doc_id):
        """Obtient la fréquence d'un terme dans un document"""
        return self.inverted_index.get(term, {}).get(doc_id, 0)
    
    def calculate_p_laplace(self, term: str, doc_id: int) -> float:
        """
        Calcule P_add-1(w|d) avec lissage de Laplace (Add-One)
        
        Formule: P_add-1(w|d) = (freq(w,d) + 1) / (|d| + V)
        
        où:
        - freq(w,d) = fréquence du terme w dans le document d
        - |d| = longueur du document (somme de toutes les fréquences)
        - V = taille du vocabulaire (nombre de termes uniques)
        
        IMPORTANT: Même si le terme n'existe pas dans le doc (freq = 0),
        la probabilité est NON NULLE: P = 1 / (|d| + V)
        
        Args:
            term: Le mot
            doc_id: ID du document
        
        Returns:
            Probabilité avec lissage de Laplace (toujours > 0)
        """
        freq_w_d = self.get_term_freq(term, doc_id)
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        # Dénominateur: |d| + V
        denominator = doc_length + self.V
        
        if denominator == 0:
            return 0.0
        
        # Numérateur: freq(w,d) + 1
        numerator = freq_w_d + 1
        
        return numerator / denominator
    
    def calculate_rsv(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calcule le RSV (Retrieval Status Value) avec lissage de Laplace
        
        RSV(Q, d) = ∏ P_add-1(w|d) pour w ∈ Q
        
        IMPORTANT: Grâce au lissage, AUCUNE probabilité n'est égale à 0
        Donc le RSV est toujours > 0 (pas de problème de zéro-fréquence)
        
        Args:
            query_terms: Liste des termes de la requête
            doc_id: ID du document
        
        Returns:
            RSV score (produit des probabilités lissées)
        """
        rsv = 1.0
        
        for term in query_terms:
            p_laplace = self.calculate_p_laplace(term, doc_id)
            rsv *= p_laplace
        
        return rsv
    
    def rank_documents(self, query_terms: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Classe tous les documents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            top_k: Si spécifié, limite le nombre de résultats (pour affichage)
                   Si None, retourne TOUS les documents (requis pour évaluation)
        
        Returns:
            Liste de tuples (doc_id, rsv_score) triée par score décroissant
        """
        # Calculer RSV pour tous les documents
        doc_scores = []
        
        for doc_id in self.doc_ids:
            rsv = self.calculate_rsv(query_terms, doc_id)
            doc_scores.append((doc_id, rsv))
        
        # Trier par score décroissant
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k si spécifié (pour affichage uniquement)
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def search(self, query_terms: List[str], top_k: int = 10, 
              verbose: bool = False, return_all: bool = False) -> List[int]:
        """
        Recherche les documents pertinents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            top_k: Nombre de documents à retourner (ignoré si return_all=True)
            verbose: Afficher les résultats
            return_all: Si True, retourne TOUS les documents (pour évaluation)
        
        Returns:
            Liste des doc_ids classés par pertinence
        """
        # Obtenir tous les documents classés
        doc_scores = self.rank_documents(query_terms, top_k=None)
        
        if verbose and doc_scores:
            display_k = min(top_k, len(doc_scores))
            print(f"\n🔍 Top {display_k} documents:")
            print(f"{'Rang':<6} {'Doc ID':<10} {'RSV Score':<15}")
            print("-" * 35)
            for rank, (doc_id, score) in enumerate(doc_scores[:display_k], 1):
                print(f"{rank:<6} {doc_id:<10} {score:.10f}")
        
        # Extraire les doc_ids
        ranked_list = [doc_id for doc_id, score in doc_scores]
        
        # Limiter seulement si demandé ET pas return_all
        if top_k and not return_all:
            ranked_list = ranked_list[:top_k]
        
        return ranked_list


# ============================================================================
# ÉVALUATION COMPLÈTE AVEC METRICS.PY
# ============================================================================

if __name__ == "__main__":
    
    # Ajouter le dossier src et evaluation au path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    src_dir = os.path.join(project_dir, 'src')
    eval_dir = os.path.join(project_dir, 'evaluation')
    sys.path.insert(0, src_dir)
    sys.path.insert(0, eval_dir)
    
    from medline_parser import parse_med_qry, parse_med_rel
    from preprocessing import MEDLINEPreprocessor
    from metrics import IRMetrics
    
    # Chemins
    INVERTED_INDEX_PATH = os.path.join(project_dir, "data", "ouput", "inverted_index.txt")
    DOC_TERM_MATRIX_PATH = os.path.join(project_dir, "data", "ouput", "document_term_matrix.txt")
    MED_QRY_PATH = os.path.join(project_dir, "data", "MED.QRY")
    MED_REL_PATH = os.path.join(project_dir, "data", "MED.REL")
    
    # Vérifier les fichiers
    for path, name in [(INVERTED_INDEX_PATH, "Inverted Index"),
                       (DOC_TERM_MATRIX_PATH, "Document-Term Matrix"),
                       (MED_QRY_PATH, "MED.QRY"),
                       (MED_REL_PATH, "MED.REL")]:
        if not os.path.exists(path):
            print(f"❌ ERREUR: Fichier non trouvé: {path}")
            exit(1)
    
    print("="*80)
    print("ÉVALUATION COMPLÈTE - LANGUAGE MODEL LAPLACE (ADD-ONE SMOOTHING)")
    print("="*80)
    
    # 1. Créer le modèle
    print("\n📚 Étape 1: Initialisation du modèle Language Model Laplace")
    print("   Technique: Add-One Smoothing (Laplace)")
    print("   Formule: P_add-1(w|d) = (freq(w,d) + 1) / (|d| + V)")
    lm_laplace = LanguageModelLaplace(
        inverted_index_path=INVERTED_INDEX_PATH,
        doc_term_matrix_path=DOC_TERM_MATRIX_PATH
    )
    print(f"✅ Modèle initialisé")
    print(f"   - Documents: {lm_laplace.N}")
    print(f"   - Vocabulaire (V): {lm_laplace.V} termes")
    
    # 2. Charger les données
    print("\n📄 Étape 2: Chargement des données")
    queries = parse_med_qry(MED_QRY_PATH)
    relevance_judgments = parse_med_rel(MED_REL_PATH)
    print(f"✅ {len(queries)} requêtes chargées")
    print(f"✅ {len(relevance_judgments)} jugements de pertinence chargés")
    
    # 3. Créer le preprocessor
    preprocessor = MEDLINEPreprocessor()
    
    # 4. Initialiser le système de métriques
    print("\n📊 Étape 3: Initialisation du système d'évaluation")
    metrics = IRMetrics(relevance_judgments, model_name="LM_Laplace")
    
    # 5. Collecter tous les résultats
    print("\n🔍 Étape 4: Traitement de toutes les requêtes")
    results_per_query = {}
    relevance_scores_per_query = {}
    
    for query in queries:
        query_id = query.query_id
        query_text = query.text
        
        # Preprocesser la requête
        query_terms = preprocessor.preprocess_text(query_text)
        
        # Obtenir TOUS les documents classés
        doc_scores = lm_laplace.rank_documents(query_terms, top_k=None)
        
        # Extraire les doc_ids et les scores
        ranked_list = [doc_id for doc_id, score in doc_scores]
        scores_dict = {doc_id: score for doc_id, score in doc_scores}
        
        results_per_query[query_id] = ranked_list
        relevance_scores_per_query[query_id] = scores_dict
        
        print(f"   Requête {query_id}: {len(ranked_list)} documents classés")
    
    # 6. Évaluer le système complet
    print("\n📈 Étape 5: Évaluation complète du système")
    all_results = metrics.evaluate_all_queries(
        results_per_query=results_per_query,
        relevance_scores_per_query=relevance_scores_per_query,
        plot_curves=True,
        save_results=True,
        verbose=True  # ✅ CHANGEMENT: verbose=True pour sauvegarder les métriques globales
    )
    
    print("\n" + "="*80)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*80)
    print(f"📁 Résultats sauvegardés:")
    print(f"   - results/LM_Laplace_results.txt")
    print(f"   - results/figures/LM_Laplace/")
    print("="*80)