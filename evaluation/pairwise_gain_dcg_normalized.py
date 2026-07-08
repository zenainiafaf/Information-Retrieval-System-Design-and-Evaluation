"""
Calcul du Gain (%) DEUX À DEUX - VERSION DCG@20
Utilise DCG@20 au lieu de nDCG@20 pour avoir des différences réelles
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ==================== EXTRACTION DCG@20 ====================

def extract_dcg_from_file(filepath: str, query_ids: List[int] = None) -> Dict[int, float]:
    """
    Extrait les scores DCG@20 (pas nDCG@20) d'un fichier de résultats
    
    Args:
        filepath: Chemin vers le fichier *_results.txt
        query_ids: Liste des IDs de requêtes à extraire (défaut: 1-10)
        
    Returns:
        Dict {query_id: dcg_score}
    """
    if query_ids is None:
        query_ids = list(range(1, 11))
    
    dcg_scores = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Séparer en blocs de requêtes
        query_blocks = re.split(r'={5,}\s*REQUÊTE\s+(\d+)\s*={5,}', content)
        
        for i in range(1, len(query_blocks), 2):
            if i + 1 < len(query_blocks):
                query_id = int(query_blocks[i])
                block_content = query_blocks[i + 1]
                
                if query_id in query_ids:
                    # Chercher DCG@20 dans la section MÉTRIQUES
                    metrics_match = re.search(r'MÉTRIQUES:\s*-+\s*(.*?)(?:CLASSEMENT|$)', 
                                             block_content, re.DOTALL)
                    
                    if metrics_match:
                        metrics_section = metrics_match.group(1)
                        
                        # Chercher DCG@20 (pas nDCG@20!)
                        dcg_match = re.search(r'DCG@20:\s+([\d.]+)', metrics_section)
                        
                        if dcg_match:
                            dcg = float(dcg_match.group(1))
                            dcg_scores[query_id] = dcg
    
    except Exception as e:
        print(f"   ⚠️  Erreur: {e}")
    
    return dcg_scores


def extract_model_name(filename: str) -> str:
    """Extrait le nom du modèle depuis le nom du fichier"""
    name = filename.replace('_results.txt', '')
    
    mappings = {
        'VSM_cosine': 'VSM',
        'VSM_Cosine': 'VSM',
        'LSI_k100': 'LSI',
        'BIR_Avec_Apprentissage': 'BIR_Classic_WithRel',
        'BIR_Sans_Apprentissage': 'BIR_Classic',
        'BIR_Extended_Avec_Apprentissage': 'BIR_Extended_WithRel',
        'BIR_Extended_Sans_Apprentissage': 'BIR_Extended',
        'bm25': 'BM25',
        'LM_Dirichlet_Auto': 'LM_Dirichlet',
        'LM_JelinekMercer_02': 'LM_JelinekMercer',
        'LM_Laplace': 'LM_Add1',
        'LM_MLE': 'LM_MLE'
    }
    
    return mappings.get(name, name)


def find_results_directory():
    """Cherche automatiquement le dossier 'results'"""
    possible_paths = ["results", "../results", "../../results"]
    
    current = os.getcwd()
    for _ in range(4):
        test_path = os.path.join(current, "results")
        if os.path.exists(test_path):
            return test_path
        current = os.path.dirname(current)
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


def load_all_results(results_dir: str = None, query_ids: List[int] = None) -> Dict[str, Dict[int, float]]:
    """
    Charge tous les fichiers et extrait les DCG@20
    
    Args:
        results_dir: Dossier contenant les fichiers *_results.txt
        query_ids: Liste des IDs de requêtes (défaut: 1-10)
        
    Returns:
        Dict {model_name: {query_id: dcg_score}}
    """
    if query_ids is None:
        query_ids = list(range(1, 11))
    
    if results_dir is None:
        results_dir = find_results_directory()
        if results_dir is None:
            print("\n❌ Impossible de trouver le dossier 'results'!")
            return {}
    
    print("\n" + "="*150)
    print("📂 CHARGEMENT DES FICHIERS - DCG@20 NORMALISÉ")
    print("="*150)
    print(f"Dossier: {results_dir}")
    print(f"Requêtes: Q{query_ids[0]}-Q{query_ids[-1]} (I{query_ids[0]}-I{query_ids[-1]})")
    print(f"Métrique: DCG@20 avec normalisation")
    print(f"Formule: Gain = (DCG_A - DCG_B) / max(DCG_A, DCG_B) × 100")
    print("-"*150)
    
    all_dcg_data = {}
    
    if not os.path.exists(results_dir):
        print(f"❌ Le dossier '{results_dir}' n'existe pas!")
        return {}
    
    result_files = [f for f in os.listdir(results_dir) 
                   if f.endswith('_results.txt') and os.path.isfile(os.path.join(results_dir, f))]
    
    if not result_files:
        print(f"\n❌ Aucun fichier *_results.txt trouvé")
        return {}
    
    print(f"\n📄 {len(result_files)} fichiers trouvés:\n")
    
    for result_file in sorted(result_files):
        filepath = os.path.join(results_dir, result_file)
        model_name = extract_model_name(result_file)
        
        print(f"   📊 {result_file}")
        print(f"      → Modèle: {model_name}")
        
        dcg_scores = extract_dcg_from_file(filepath, query_ids)
        
        if dcg_scores:
            all_dcg_data[model_name] = dcg_scores
            num_queries = len(dcg_scores)
            mean_dcg = sum(dcg_scores.values()) / num_queries if num_queries > 0 else 0.0
            print(f"      ✅ {num_queries} scores DCG@20 extraits (Moyenne: {mean_dcg:.4f})")
            
            # Afficher quelques scores
            sample_queries = list(dcg_scores.items())[:3]
            for qid, score in sample_queries:
                print(f"         Q{qid}: DCG@20 = {score:.4f}")
        else:
            print(f"      ⚠️  Aucun score DCG@20 trouvé")
        
        print()
    
    print("-"*150)
    print(f"✅ Total: {len(all_dcg_data)} modèles chargés")
    print("="*150)
    
    return all_dcg_data


# ==================== CLASSE PAIRWISE (identique, juste renommer les variables) ====================

class PairwiseGainCalculator:
    """Classe pour calculer les gains deux à deux - VERSION DCG@20 NORMALISÉE"""
    
    def __init__(self, dcg_data: Dict[str, Dict[int, float]]):
        self.dcg_data = dcg_data
        self.model_names = sorted(dcg_data.keys())
    
    def calculate_gain(self, model_a: str, model_b: str, query_id: int) -> float:
        """
        Calcule le gain NORMALISÉ de model_a par rapport à model_b
        
        Formule: Gain = (DCG_A - DCG_B) / max(DCG_A, DCG_B) × 100
        
        Cette normalisation limite les gains entre -100% et +100%
        """
        dcg_a = self.dcg_data[model_a].get(query_id, 0.0)
        dcg_b = self.dcg_data[model_b].get(query_id, 0.0)
        
        # Normaliser par le maximum des deux
        dcg_max = max(dcg_a, dcg_b)
        
        if dcg_max == 0.0:
            return 0.0
        
        # Gain normalisé: toujours entre -100% et +100%
        gain = ((dcg_a - dcg_b) / dcg_max) * 100
        
        return gain
    
    def calculate_mean_gain(self, model_a: str, model_b: str, query_ids: List[int]) -> float:
        """Calcule le gain moyen"""
        gains = []
        for qid in query_ids:
            gain = self.calculate_gain(model_a, model_b, qid)
            if gain != float('inf') and gain != float('-inf'):
                gains.append(gain)
        return sum(gains) / len(gains) if gains else 0.0
    
    def generate_pairwise_matrix(self, query_ids: List[int] = None) -> pd.DataFrame:
        """Génère la matrice de gains"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        matrix = np.zeros((len(self.model_names), len(self.model_names)))
        
        for i, model_a in enumerate(self.model_names):
            for j, model_b in enumerate(self.model_names):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    gain = self.calculate_mean_gain(model_a, model_b, query_ids)
                    matrix[i, j] = gain
        
        return pd.DataFrame(matrix, index=self.model_names, columns=self.model_names)
    
    def generate_comparison_table(self, model_a: str, model_b: str, query_ids: List[int] = None) -> pd.DataFrame:
        """Génère un tableau de comparaison"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        data = {
            'Query': [f'Q{qid}' for qid in query_ids],
            f'{model_a} DCG@20': [self.dcg_data[model_a].get(qid, 0.0) for qid in query_ids],
            f'{model_b} DCG@20': [self.dcg_data[model_b].get(qid, 0.0) for qid in query_ids],
            f'Gain {model_a} vs {model_b} (%)': [self.calculate_gain(model_a, model_b, qid) for qid in query_ids]
        }
        
        df = pd.DataFrame(data)
        
        mean_row = {
            'Query': 'Mean',
            f'{model_a} DCG@20': df[f'{model_a} DCG@20'].mean(),
            f'{model_b} DCG@20': df[f'{model_b} DCG@20'].mean(),
            f'Gain {model_a} vs {model_b} (%)': self.calculate_mean_gain(model_a, model_b, query_ids)
        }
        
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        return df
    
    def find_best_pairs(self, query_ids: List[int] = None, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Trouve les meilleures paires"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        pairs = []
        for i, model_a in enumerate(self.model_names):
            for j, model_b in enumerate(self.model_names):
                if i != j:
                    gain = self.calculate_mean_gain(model_a, model_b, query_ids)
                    pairs.append((model_a, model_b, gain))
        
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs[:top_n]
    
    def print_pairwise_matrix(self, query_ids: List[int] = None):
        """Affiche la matrice"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        print("\n" + "="*150)
        print("📊 MATRICE DE GAINS DEUX À DEUX - DCG@20 NORMALISÉ")
        print("="*150)
        print(f"Requêtes: Q{query_ids[0]}-Q{query_ids[-1]} (I{query_ids[0]}-I{query_ids[-1]})")
        print(f"\n💡 Formule de normalisation: Gain = (DCG_A - DCG_B) / max(DCG_A, DCG_B) × 100")
        print(f"   ✅ Gains limités entre -100% et +100%")
        print("\n💡 Lecture:")
        print("   - Ligne = Modèle évalué")
        print("   - Colonne = Modèle de référence")
        print("   - Valeur positive = Ligne MEILLEUR")
        print("-"*150)
        
        matrix_df = self.generate_pairwise_matrix(query_ids)
        print(matrix_df.to_string(float_format=lambda x: f"{x:+.2f}%"))
        print("\n" + "="*150)
    
    def print_comparison(self, model_a: str, model_b: str, query_ids: List[int] = None):
        """Affiche une comparaison détaillée"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        print("\n" + "="*150)
        print(f"⚖️  COMPARAISON: {model_a} vs {model_b} (DCG@20 Normalisé)")
        print("="*150)
        
        comparison_df = self.generate_comparison_table(model_a, model_b, query_ids)
        print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        mean_gain = self.calculate_mean_gain(model_a, model_b, query_ids)
        
        print("\n" + "-"*150)
        print("📈 ANALYSE:")

        SEUIL = 5.0
        
        if mean_gain >= SEUIL :
            print(f"   ✅ {model_a} est MEILLEUR que {model_b} (+{mean_gain:.2f}%)")
        elif mean_gain < SEUIL :
            print(f"   ❌ {model_a} est MOINS BON que {model_b} ({mean_gain:.2f}%)")
        
        
        wins_a = sum(1 for qid in query_ids if self.calculate_gain(model_a, model_b, qid) > 0)
        wins_b = sum(1 for qid in query_ids if self.calculate_gain(model_a, model_b, qid) < 0)
        ties = len(query_ids) - wins_a - wins_b
        
        print(f"   📊 Victoires: {model_a}: {wins_a} | {model_b}: {wins_b} | Égalités: {ties}")
        print("="*150 + "\n")
    
    def print_top_differences(self, query_ids: List[int] = None, top_n: int = 10):
        """Affiche les top différences"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        print("\n" + "="*150)
        print(f"🔝 TOP {top_n} DIFFÉRENCES (DCG@20)")
        print("="*150)
        
        best_pairs = self.find_best_pairs(query_ids, top_n)
        
        print(f"\n{'Rang':<6} {'Modèle A':<25} {'vs':<4} {'Modèle B':<25} {'Gain (%)':<20}")
        print("-"*150)
        
        for rank, (model_a, model_b, gain) in enumerate(best_pairs, 1):
            symbol = ">" if gain > 0 else "<"
            print(f"{rank:<6} {model_a:<25} {symbol:<4} {model_b:<25} {gain:+.2f}%")
        
        print("="*150 + "\n")
    
    def save_all_results(self, output_dir: str = "results/gainDCGnormalised", query_ids: List[int] = None):
        """Sauvegarde les résultats"""
        if query_ids is None:
            query_ids = list(range(1, 11))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Matrice
        matrix_df = self.generate_pairwise_matrix(query_ids)
        matrix_file = os.path.join(output_dir, "pairwise_gain_matrix_DCG_normalized.csv")
        matrix_df.to_csv(matrix_file)
        print(f"✅ Matrice DCG normalisée: {matrix_file}")
        
        # Comparaisons
        comparisons_dir = os.path.join(output_dir, "pairwise_comparisons_DCG_normalised")
        os.makedirs(comparisons_dir, exist_ok=True)
        
        for i, model_a in enumerate(self.model_names):
            for j, model_b in enumerate(self.model_names):
                if i < j:
                    comparison_df = self.generate_comparison_table(model_a, model_b, query_ids)
                    filename = f"{model_a}_vs_{model_b}.csv"
                    filepath = os.path.join(comparisons_dir, filename)
                    comparison_df.to_csv(filepath, index=False)
        
        print(f"✅ Comparaisons DCG: {comparisons_dir}/")
        
        # Rapport
        report_file = os.path.join(output_dir, "pairwise_gain_report_DCG_normalized.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*150 + "\n")
            f.write("ANALYSE DES GAINS DEUX À DEUX - DCG@20 NORMALISÉ\n")
            f.write("="*150 + "\n\n")
            f.write("Formule: Gain = (DCG_A - DCG_B) / max(DCG_A, DCG_B) × 100\n")
            f.write("Gains limités entre -100% et +100%\n\n")
            f.write("MATRICE:\n")
            f.write("-"*150 + "\n")
            f.write(matrix_df.to_string(float_format=lambda x: f"{x:+.2f}%"))
            f.write("\n\n")
            f.write("TOP 20:\n")
            f.write("-"*150 + "\n")
            best_pairs = self.find_best_pairs(query_ids, 20)
            f.write(f"{'Rang':<6} {'Modèle A':<25} {'vs':<4} {'Modèle B':<25} {'Gain':<20}\n")
            f.write("-"*150 + "\n")
            for rank, (model_a, model_b, gain) in enumerate(best_pairs, 1):
                symbol = ">" if gain > 0 else "<"
                f.write(f"{rank:<6} {model_a:<25} {symbol:<4} {model_b:<25} {gain:+.2f}%\n")
            f.write("\n" + "="*150 + "\n")
        
        print(f"✅ Rapport DCG normalisé: {report_file}")


# ==================== FONCTIONS PRINCIPALES ====================

def pairwise_analysis_from_files(results_dir: str = None, query_ids: List[int] = None):
    """Analyse complète avec DCG@20 normalisé"""
    if query_ids is None:
        query_ids = list(range(1, 11))
    
    print("\n" + "="*150)
    print("🔄 ANALYSE DEUX À DEUX - DCG@20 NORMALISÉ (I1-I10)")
    print("="*150)
    
    dcg_data = load_all_results(results_dir, query_ids)
    
    if not dcg_data:
        print("\n❌ Aucune donnée chargée!")
        return None
    
    calc = PairwiseGainCalculator(dcg_data)
    calc.print_pairwise_matrix(query_ids)
    calc.print_top_differences(query_ids, top_n=15)
    calc.save_all_results(output_dir="results/gainDCGnrmalised", query_ids=query_ids)
    
    return calc


def compare_two_models(model_a: str, model_b: str, results_dir: str = None, query_ids: List[int] = None):
    """Compare deux modèles avec DCG@20"""
    if query_ids is None:
        query_ids = list(range(1, 11))
    
    dcg_data = load_all_results(results_dir, query_ids)
    
    if not dcg_data:
        print("\n❌ Aucune donnée chargée!")
        return
    
    if model_a not in dcg_data:
        print(f"\n❌ Modèle '{model_a}' non trouvé!")
        print(f"   Disponibles: {', '.join(dcg_data.keys())}")
        return
    
    if model_b not in dcg_data:
        print(f"\n❌ Modèle '{model_b}' non trouvé!")
        print(f"   Disponibles: {', '.join(dcg_data.keys())}")
        return
    
    calc = PairwiseGainCalculator(dcg_data)
    calc.print_comparison(model_a, model_b, query_ids)


# ==================== EXÉCUTION ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if len(sys.argv) >= 3:
            model_a = sys.argv[1]
            model_b = sys.argv[2]
            print(f"\n🔍 Comparaison DCG@20 Normalisé: {model_a} vs {model_b}")
            compare_two_models(model_a, model_b)
        else:
            print("Usage: python pairwise_gain_dcg.py [MODEL_A MODEL_B]")
    else:
        print("\n🔄 Analyse complète avec DCG@20 Normalisé (I1-I10)")
        pairwise_analysis_from_files()