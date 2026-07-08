import numpy as np
from collections import defaultdict
import os
import sys
from typing import List, Dict, Tuple


class VSMModel:
    """Vector Space Model (VSM) avec similarité cosinus"""
    
    def __init__(self):
        # Vocabulaire et documents
        self.vocabulary = []      # Liste ordonnée des termes
        self.doc_ids = []         # Liste ordonnée des doc_ids
        
        # Matrice TF-IDF (M × N)
        self.doc_vectors = None   # Vecteurs TF-IDF des documents
        
        # Index inversé pour accès rapide
        self.inverted_index = {}  # {term: {doc_id: tfidf_weight}}
    
    
    def load_inverted_index(self, filepath: str, verbose: bool = True):
        """Charge l'inverted index et construit la matrice TF-IDF"""
        if verbose:
            print("\n" + "="*80)
            print("CHARGEMENT DE L'INVERTED INDEX")
            print("="*80)
        
        # Structure: {term: {doc_id: weight}}
        term_doc_weights = defaultdict(dict)
        all_docs = set()
        all_terms = set()
        
        # Lire le fichier
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    term = parts[0]
                    doc_id = int(parts[1])
                    weight = float(parts[3])  # TF-IDF weight
                    
                    term_doc_weights[term][doc_id] = weight
                    all_terms.add(term)
                    all_docs.add(doc_id)
        
        # Créer les listes ordonnées
        self.vocabulary = sorted(list(all_terms))
        self.doc_ids = sorted(list(all_docs))
        
        M = len(self.vocabulary)  # Nombre de termes
        N = len(self.doc_ids)     # Nombre de documents
        
        if verbose:
            print(f"\n📊 Statistiques:")
            print(f"   - Termes dans le vocabulaire: {M}")
            print(f"   - Documents: {N}")
        
        # Construire la matrice TF-IDF (M × N)
        self.doc_vectors = np.zeros((M, N), dtype=np.float32)
        
        for i, term in enumerate(self.vocabulary):
            for j, doc_id in enumerate(self.doc_ids):
                if doc_id in term_doc_weights[term]:
                    self.doc_vectors[i, j] = term_doc_weights[term][doc_id]
        
        # Stocker l'index inversé
        self.inverted_index = {term: dict(term_doc_weights[term]) 
                              for term in self.vocabulary}
        
        if verbose:
            non_zero = np.count_nonzero(self.doc_vectors)
            density = (non_zero / (M * N)) * 100
            print(f"\n✅ Matrice document-terme construite: {M} × {N}")
            print(f"   - Éléments non-nuls: {non_zero:,}")
            print(f"   - Densité: {density:.2f}%")
            print("="*80)
    
    
    def create_query_vector(self, query_terms: List[str]) -> np.ndarray:
        """Crée le vecteur de requête (pondération binaire)"""
        # Créer le vecteur de requête (présence binaire)
        query_vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        
        found_terms = []
        for term in query_terms:
            if term in self.vocabulary:
                idx = self.vocabulary.index(term)
                query_vector[idx] = 1.0  # Pondération binaire
                found_terms.append(term)
        
        if len(found_terms) == 0:
            return None
        
        return query_vector
    
    
    def compute_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """Calcule la similarité cosinus entre la requête et tous les documents"""
        # Produit scalaire query · documents
        dot_products = query_vector @ self.doc_vectors  # (N,)
        
        # Norme de la requête
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(self.doc_ids))
        
        # Normes des documents
        doc_norms = np.linalg.norm(self.doc_vectors, axis=0)  # (N,)
        
        # Éviter division par zéro
        doc_norms[doc_norms == 0] = 1
        
        # Similarité cosinus
        similarities = dot_products / (query_norm * doc_norms)
        
        return similarities
    
    
    def rank_documents(self, query_terms: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Classe tous les documents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            top_k: Si spécifié, limite le nombre de résultats (pour affichage)
                   Si None, retourne TOUS les documents (requis pour évaluation)
        
        Returns:
            Liste de tuples (doc_id, cosine_similarity) triée par score décroissant
        """
        # Créer le vecteur de requête
        query_vector = self.create_query_vector(query_terms)
        
        if query_vector is None:
            # Aucun terme trouvé, retourner tous les docs avec score 0
            return [(doc_id, 0.0) for doc_id in self.doc_ids]
        
        # Calculer les similarités cosinus
        similarities = self.compute_cosine_similarity(query_vector)
        
        # Créer la liste (doc_id, score)
        doc_scores = [(self.doc_ids[i], float(similarities[i])) 
                     for i in range(len(self.doc_ids))]
        
        # Trier par score décroissant
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k si spécifié (pour affichage uniquement)
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    
    def fit(self, inverted_index_path: str, verbose: bool = True):
        """
        Initialise le modèle VSM
        
        Args:
            inverted_index_path: Chemin vers l'inverted index
            verbose: Afficher les détails
        """
        if verbose:
            print("\n" + "="*80)
            print("INITIALISATION DU MODÈLE VSM")
            print("="*80)
            print("Mesure de similarité: Cosine")
        
        # Charger l'inverted index
        self.load_inverted_index(inverted_index_path, verbose)
        
        if verbose:
            print("\n" + "="*80)
            print("✅ MODÈLE VSM PRÊT")
            print("="*80)
    
    
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
            print(f"{'Rang':<6} {'Doc ID':<10} {'Score':<12}")
            print("-" * 30)
            for rank, (doc_id, score) in enumerate(doc_scores[:display_k], 1):
                print(f"{rank:<6} {doc_id:<10} {score:.6f}")
        
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
    MED_QRY_PATH = os.path.join(project_dir, "data", "MED.QRY")
    MED_REL_PATH = os.path.join(project_dir, "data", "MED.REL")
    
    # Vérifier les fichiers
    for path, name in [(INVERTED_INDEX_PATH, "Inverted Index"),
                       (MED_QRY_PATH, "MED.QRY"),
                       (MED_REL_PATH, "MED.REL")]:
        if not os.path.exists(path):
            print(f"❌ ERREUR: Fichier non trouvé: {path}")
            exit(1)
    
    print("="*80)
    print("ÉVALUATION COMPLÈTE DU MODÈLE VSM (Cosine Similarity)")
    print("="*80)
    
    # 1. Créer et charger le modèle
    print("\n📚 Étape 1: Initialisation du modèle VSM")
    vsm = VSMModel()
    vsm.fit(INVERTED_INDEX_PATH, verbose=True)
    
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
    metrics = IRMetrics(relevance_judgments, model_name="VSM_Cosine")
    
    # 5. Collecter tous les résultats
    print("\n🔍 Étape 4: Traitement de toutes les requêtes")
    results_per_query = {}
    relevance_scores_per_query = {}
    
    for query in queries:
        query_id = query.query_id
        query_text = query.text
        
        # Preprocesser la requête
        query_terms = preprocessor.preprocess_text(query_text)
        
        # ✅ CRITICAL: Obtenir TOUS les documents classés (pas de limitation top_k)
        doc_scores = vsm.rank_documents(query_terms, top_k=None)
        
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
        relevance_scores_per_query=relevance_scores_per_query,  # Pour DCG/nDCG
        plot_curves=True,
        save_results=True,
        verbose=False  # Mettre True pour voir les détails de chaque requête
    )
    
    print("\n" + "="*80)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*80)
    print(f"📁 Résultats sauvegardés:")
    print(f"   - results/VSM_Cosine_results.txt")
    print(f"   - results/figures/VSM_Cosine/")
    print("="*80)
    
    # 7. Afficher un exemple détaillé (Requête 1)
    print("\n" + "="*80)
    print("📊 EXEMPLE DÉTAILLÉ - REQUÊTE 1")
    print("="*80)
    
    query_1 = queries[0]
    query_terms = preprocessor.preprocess_text(query_1.text)
    
    print(f"\nTexte: {query_1.text[:80]}...")
    print(f"Termes: {' '.join(query_terms[:10])}...")
    
    # Afficher le top 20 avec détails
    doc_scores = vsm.rank_documents(query_terms, top_k=20)
    relevant_docs = set(relevance_judgments.get(1, []))
    
    print(f"\n{'Rang':<6} {'Doc ID':<10} {'Cosine Similarity':<20} {'Pertinent':<12}")
    print("-" * 60)
    
    for rank, (doc_id, score) in enumerate(doc_scores, 1):
        is_relevant = "✓" if doc_id in relevant_docs else "✗"
        print(f"{rank:<6} {doc_id:<10} {score:<20.6f} {is_relevant:<12}")
    
    print("\n" + "="*80)