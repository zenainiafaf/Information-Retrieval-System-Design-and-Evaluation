import math
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple


class BM25Model:
    """Modèle BM25 (Best Matching 25) pour la recherche d'information"""
    
    def __init__(self, inverted_index_path, doc_term_matrix_path, k1=1.2, b=0.75):
        """
        Args:
            inverted_index_path: Chemin vers l'inverted index
            doc_term_matrix_path: Chemin vers la matrice document-terme
            k1: Paramètre de saturation de la fréquence des termes (défaut: 1.2)
            b: Paramètre de normalisation de longueur (défaut: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Charger l'index inversé
        self.inverted_index = self._load_inverted_index(inverted_index_path)
        
        # Charger les longueurs des documents
        self.doc_lengths = self._load_doc_lengths(doc_term_matrix_path)
        
        # Calculer les statistiques
        self.N = len(self.doc_lengths)  # Nombre de documents
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.N if self.N > 0 else 0
        
        # Liste des doc_ids
        self.doc_ids = sorted(list(self.doc_lengths.keys()))
    
    def _load_inverted_index(self, filepath):
        """Charge l'inverted index depuis le fichier"""
        inverted_index = defaultdict(dict)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    term = parts[0]
                    doc_id = int(parts[1])
                    freq = int(parts[2])
                    inverted_index[term][doc_id] = freq
        return inverted_index
    
    def _load_doc_lengths(self, filepath):
        """Charge les longueurs des documents depuis la matrice document-terme"""
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
    
    def compute_ni(self, term):
        """Calcule le nombre de documents contenant le terme"""
        return len(self.inverted_index.get(term, {}))
    
    def calculate_rsv(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calcule le score RSV (Retrieval Status Value) BM25 pour un document
        
        RSV = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        
        où:
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
        - f(qi, D) = fréquence du terme qi dans le document D
        - |D| = longueur du document D
        - avgdl = longueur moyenne des documents
        """
        rsv = 0.0
        dl = self.doc_lengths.get(doc_id, 0)
        
        if dl == 0:
            return 0.0
        
        for term in query_terms:
            tf = self.get_term_freq(term, doc_id)
            
            if tf > 0:
                ni = self.compute_ni(term)
                if ni == 0:
                    continue
                
                # IDF
                idf = math.log10((self.N - ni + 0.5) / (ni + 0.5))
                
                # Normalisation de longueur
                normalization = 1 - self.b + self.b * (dl / self.avg_doc_length)
                
                # Composante TF
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * normalization)
                
                # RSV
                rsv += idf * tf_component
        
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
            print(f"{'Rang':<6} {'Doc ID':<10} {'RSV Score':<12}")
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
    print("ÉVALUATION COMPLÈTE DU MODÈLE BM25")
    print("="*80)
    
    # 1. Créer le modèle BM25
    print("\n📚 Étape 1: Initialisation du modèle BM25")
    print(f"   Paramètres: k1=1.2, b=0.75")
    bm25 = BM25Model(
        inverted_index_path=INVERTED_INDEX_PATH,
        doc_term_matrix_path=DOC_TERM_MATRIX_PATH,
        k1=1.2,
        b=0.75
    )
    print(f"✅ Modèle initialisé")
    print(f"   - Documents: {bm25.N}")
    print(f"   - Longueur moyenne: {bm25.avg_doc_length:.2f} termes")
    print(f"   - Vocabulaire: {len(bm25.inverted_index)} termes")
    
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
    metrics = IRMetrics(relevance_judgments, model_name="BM25")
    
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
        doc_scores = bm25.rank_documents(query_terms, top_k=None)
        
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
    print(f"   - results/BM25_results.txt")
    print(f"   - results/figures/BM25/")
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
    doc_scores = bm25.rank_documents(query_terms, top_k=20)
    relevant_docs = set(relevance_judgments.get(1, []))
    
    print(f"\n{'Rang':<6} {'Doc ID':<10} {'RSV Score':<15} {'Pertinent':<12}")
    print("-" * 50)
    
    for rank, (doc_id, score) in enumerate(doc_scores, 1):
        is_relevant = "✓" if doc_id in relevant_docs else "✗"
        print(f"{rank:<6} {doc_id:<10} {score:<15.6f} {is_relevant:<12}")
    
    print("\n" + "="*80)