import math
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple


class ClassicBIRModel:
    """Modèle BIR Classique (Binary Independence Retrieval)"""
    
    def __init__(self, inverted_index_path, doc_term_matrix_path):
        """
        Args:
            inverted_index_path: Chemin vers l'inverted index
            doc_term_matrix_path: Chemin vers la matrice document-terme
        """
        # Charger l'index inversé
        self.inverted_index, self.binary_matrix = self._load_inverted_index(inverted_index_path)
        
        # Charger les longueurs des documents (pour avoir la liste complète des doc_ids)
        self.doc_lengths = self._load_doc_lengths(doc_term_matrix_path)
        
        # Calculer les statistiques
        self.N = len(self.doc_lengths)  # Nombre de documents
        
        # Liste des doc_ids
        self.doc_ids = sorted(list(self.doc_lengths.keys()))
    
    def _load_inverted_index(self, filepath):
        """Charge l'inverted index depuis le fichier"""
        inverted_index = defaultdict(dict)
        binary_matrix = defaultdict(lambda: defaultdict(int))
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    term = parts[0]
                    doc_id = int(parts[1])
                    freq = int(parts[2])
                    weight = float(parts[3])
                    
                    inverted_index[term][doc_id] = {'freq': freq, 'weight': weight}
                    binary_matrix[term][doc_id] = 1  # Présence binaire
        
        return dict(inverted_index), dict(binary_matrix)
    
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
    
    def compute_ni(self, term):
        """Calcule le nombre de documents contenant le terme"""
        if term not in self.binary_matrix:
            return 0
        return len(self.binary_matrix[term])
    
    def compute_ri(self, term, relevant_docs):
        """Calcule le nombre de documents pertinents contenant le terme"""
        if term not in self.binary_matrix:
            return 0
        
        r = 0
        for doc_id in relevant_docs:
            if doc_id in self.binary_matrix[term]:
                r += 1
        return r
    
    def calculate_rsv_without_relevance(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calcule le score RSV BIR SANS données d'apprentissage
        
        RSV(q,d) = Σ log10((N - n_i + 0.5) / (n_i + 0.5))
        
        où:
        - N = nombre total de documents
        - n_i = nombre de documents contenant le terme i
        """
        rsv = 0.0
        
        for term in query_terms:
            # Vérifier si le terme est dans le document
            if term in self.binary_matrix and doc_id in self.binary_matrix[term]:
                # Calculer n_i
                ni = self.compute_ni(term)
                
                # Formule BIR sans apprentissage
                numerator = self.N - ni + 0.5
                denominator = ni + 0.5
                
                rsv += math.log10(numerator / denominator)
        
        return rsv
    
    def calculate_rsv_with_relevance(self, query_terms: List[str], doc_id: int, 
                                     relevant_docs: List[int]) -> float:
        """
        Calcule le score RSV BIR AVEC données d'apprentissage
        
        RSV(q,d) = Σ log10((r_i + 0.5)(N - R - n_i + r_i + 0.5) / 
                           ((n_i - r_i + 0.5)(R - r_i + 0.5)))
        
        où:
        - r_i = nombre de docs pertinents contenant le terme i
        - R = nombre total de docs pertinents
        - n_i = nombre de docs contenant le terme i
        - N = nombre total de docs
        """
        rsv = 0.0
        R = len(relevant_docs)
        
        for term in query_terms:
            # Vérifier si le terme est dans le document
            if term in self.binary_matrix and doc_id in self.binary_matrix[term]:
                # Calculer n_i et r_i
                ni = self.compute_ni(term)
                ri = self.compute_ri(term, relevant_docs)
                
                # Formule BIR avec apprentissage
                numerator = (ri + 0.5) * (self.N - R - ni + ri + 0.5)
                denominator = (ni - ri + 0.5) * (R - ri + 0.5)
                
                if denominator > 0:
                    rsv += math.log10(numerator / denominator)
        
        return rsv
    
    def rank_documents(self, query_terms: List[str], relevant_docs: List[int] = None, 
                      top_k: int = None) -> List[Tuple[int, float]]:
        """
        Classe tous les documents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            relevant_docs: Documents pertinents (pour BIR avec apprentissage)
                          Si None, utilise BIR sans apprentissage
            top_k: Si spécifié, limite le nombre de résultats (pour affichage)
                   Si None, retourne TOUS les documents (requis pour évaluation)
        
        Returns:
            Liste de tuples (doc_id, rsv_score) triée par score décroissant
        """
        # Calculer RSV pour tous les documents
        doc_scores = []
        
        for doc_id in self.doc_ids:
            if relevant_docs is not None:
                # BIR AVEC apprentissage
                rsv = self.calculate_rsv_with_relevance(query_terms, doc_id, relevant_docs)
            else:
                # BIR SANS apprentissage
                rsv = self.calculate_rsv_without_relevance(query_terms, doc_id)
            
            doc_scores.append((doc_id, rsv))
        
        # Trier par score décroissant
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top_k si spécifié (pour affichage uniquement)
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def search(self, query_terms: List[str], relevant_docs: List[int] = None, 
              top_k: int = 10, verbose: bool = False, return_all: bool = False) -> List[int]:
        """
        Recherche les documents pertinents pour une requête
        
        Args:
            query_terms: Liste des termes de la requête
            relevant_docs: Documents pertinents (pour BIR avec apprentissage)
            top_k: Nombre de documents à retourner (ignoré si return_all=True)
            verbose: Afficher les résultats
            return_all: Si True, retourne TOUS les documents (pour évaluation)
        
        Returns:
            Liste des doc_ids classés par pertinence
        """
        # Obtenir tous les documents classés
        doc_scores = self.rank_documents(query_terms, relevant_docs, top_k=None)
        
        if verbose and doc_scores:
            display_k = min(top_k, len(doc_scores))
            mode = "AVEC apprentissage" if relevant_docs else "SANS apprentissage"
            print(f"\n🔍 Top {display_k} documents ({mode}):")
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
    print("ÉVALUATION COMPLÈTE DU MODÈLE BIR CLASSIQUE")
    print("="*80)
    
    # 1. Créer le modèle BIR
    print("\n📚 Étape 1: Initialisation du modèle BIR")
    bir = ClassicBIRModel(
        inverted_index_path=INVERTED_INDEX_PATH,
        doc_term_matrix_path=DOC_TERM_MATRIX_PATH
    )
    print(f"✅ Modèle initialisé")
    print(f"   - Documents: {bir.N}")
    print(f"   - Vocabulaire: {len(bir.inverted_index)} termes")
    
    # 2. Charger les données
    print("\n📄 Étape 2: Chargement des données")
    queries = parse_med_qry(MED_QRY_PATH)
    relevance_judgments = parse_med_rel(MED_REL_PATH)
    print(f"✅ {len(queries)} requêtes chargées")
    print(f"✅ {len(relevance_judgments)} jugements de pertinence chargés")
    
    # 3. Créer le preprocessor
    preprocessor = MEDLINEPreprocessor()
    
    # 4. Initialiser le système de métriques (SANS apprentissage)
    print("\n📊 Étape 3: Initialisation du système d'évaluation")
    metrics_without = IRMetrics(relevance_judgments, model_name="BIR_Sans_Apprentissage")
    
    # 5. Collecter tous les résultats SANS apprentissage
    print("\n🔍 Étape 4: Traitement de toutes les requêtes (SANS apprentissage)")
    results_without = {}
    scores_without = {}
    
    for query in queries:
        query_id = query.query_id
        query_text = query.text
        
        # Preprocesser la requête
        query_terms = preprocessor.preprocess_text(query_text)
        
        # ✅ CRITICAL: Obtenir TOUS les documents classés (pas de limitation top_k)
        doc_scores = bir.rank_documents(query_terms, relevant_docs=None, top_k=None)
        
        # Extraire les doc_ids et les scores
        ranked_list = [doc_id for doc_id, score in doc_scores]
        scores_dict = {doc_id: score for doc_id, score in doc_scores}
        
        results_without[query_id] = ranked_list
        scores_without[query_id] = scores_dict
        
        print(f"   Requête {query_id}: {len(ranked_list)} documents classés")
    
    # 6. Évaluer le système complet SANS apprentissage
    print("\n📈 Étape 5: Évaluation complète (SANS apprentissage)")
    all_results_without = metrics_without.evaluate_all_queries(
        results_per_query=results_without,
        relevance_scores_per_query=scores_without,
        plot_curves=True,
        save_results=True,
        verbose=False
    )
    
    # 7. Initialiser le système de métriques (AVEC apprentissage)
    print("\n📊 Étape 6: Initialisation pour BIR AVEC apprentissage")
    metrics_with = IRMetrics(relevance_judgments, model_name="BIR_Avec_Apprentissage")
    
    # 8. Collecter tous les résultats AVEC apprentissage
    print("\n🔍 Étape 7: Traitement de toutes les requêtes (AVEC apprentissage)")
    results_with = {}
    scores_with = {}
    
    for query in queries:
        query_id = query.query_id
        query_text = query.text
        
        # Preprocesser la requête
        query_terms = preprocessor.preprocess_text(query_text)
        
        # Obtenir les documents pertinents
        relevant_docs = relevance_judgments.get(query_id, [])
        
        # ✅ CRITICAL: Obtenir TOUS les documents classés AVEC apprentissage
        doc_scores = bir.rank_documents(query_terms, relevant_docs=relevant_docs, top_k=None)
        
        # Extraire les doc_ids et les scores
        ranked_list = [doc_id for doc_id, score in doc_scores]
        scores_dict = {doc_id: score for doc_id, score in doc_scores}
        
        results_with[query_id] = ranked_list
        scores_with[query_id] = scores_dict
        
        print(f"   Requête {query_id}: {len(ranked_list)} documents classés (R={len(relevant_docs)})")
    
    # 9. Évaluer le système complet AVEC apprentissage
    print("\n📈 Étape 8: Évaluation complète (AVEC apprentissage)")
    all_results_with = metrics_with.evaluate_all_queries(
        results_per_query=results_with,
        relevance_scores_per_query=scores_with,
        plot_curves=True,
        save_results=True,
        verbose=False
    )
    
    print("\n" + "="*80)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*80)
    print(f"📁 Résultats sauvegardés:")
    print(f"   - results/BIR_Sans_Apprentissage_results.txt")
    print(f"   - results/figures/BIR_Sans_Apprentissage/")
    print(f"   - results/BIR_Avec_Apprentissage_results.txt")
    print(f"   - results/figures/BIR_Avec_Apprentissage/")
    print("="*80)
    
    # 10. Afficher un exemple détaillé (Requête 1)
    print("\n" + "="*80)
    print("📊 EXEMPLE DÉTAILLÉ - REQUÊTE 1")
    print("="*80)
    
    query_1 = queries[0]
    query_terms = preprocessor.preprocess_text(query_1.text)
    
    print(f"\nTexte: {query_1.text[:80]}...")
    print(f"Termes: {' '.join(query_terms[:10])}...")
    
    # SANS apprentissage
    print(f"\n{'='*80}")
    print("BIR SANS APPRENTISSAGE")
    print("="*80)
    doc_scores = bir.rank_documents(query_terms, relevant_docs=None, top_k=20)
    relevant_docs_set = set(relevance_judgments.get(1, []))
    
    print(f"\n{'Rang':<6} {'Doc ID':<10} {'RSV Score':<15} {'Pertinent':<12}")
    print("-" * 50)
    
    for rank, (doc_id, score) in enumerate(doc_scores, 1):
        is_relevant = "✓" if doc_id in relevant_docs_set else "✗"
        print(f"{rank:<6} {doc_id:<10} {score:<15.6f} {is_relevant:<12}")
    
    # AVEC apprentissage
    print(f"\n{'='*80}")
    print("BIR AVEC APPRENTISSAGE")
    print("="*80)
    print(f"Documents pertinents utilisés: {list(relevant_docs_set)[:10]}...")
    
    doc_scores_with = bir.rank_documents(query_terms, relevant_docs=list(relevant_docs_set), top_k=20)
    
    print(f"\n{'Rang':<6} {'Doc ID':<10} {'RSV Score':<15} {'Pertinent':<12}")
    print("-" * 50)
    
    for rank, (doc_id, score) in enumerate(doc_scores_with, 1):
        is_relevant = "✓" if doc_id in relevant_docs_set else "✗"
        print(f"{rank:<6} {doc_id:<10} {score:<15.6f} {is_relevant:<12}")
    
    print("\n" + "="*80)