import math
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class IRMetrics:
    """Classe pour calculer et visualiser les métriques d'évaluation des systèmes de RI"""
    
    def __init__(self, relevance_judgments: Dict[int, List[int]], model_name: str = "IR_Model"):
        """
        Args:
            relevance_judgments: Dict {query_id: [liste des doc_ids pertinents]}
            model_name: Nom du modèle pour les dossiers de sortie
        """
        self.relevance_judgments = relevance_judgments
        self.model_name = model_name
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Crée la structure de dossiers pour les résultats"""
        base_dir = f"results/figures/{self.model_name}"
        self.dirs = {
            'base': base_dir,
            'curves_non_interpolated': os.path.join(base_dir, "curves_non_interpolated"),
            'curves_interpolated': os.path.join(base_dir, "curves_interpolated"),
            'all_queries': base_dir
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # ==================== MÉTRIQUES DE BASE ====================
    
    def precision(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        Precision = Nombre de documents pertinents sélectionnés / 
                    Nombre total de documents sélectionnés
        """
        if len(retrieved) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(retrieved)
    
    def recall(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        Recall = Nombre de documents pertinents sélectionnés / 
                 Nombre total de documents pertinents
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
        F1 = 2 * P * R / (P + R)
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    # ==================== COURBES PRÉCISION-RAPPEL ====================
    
    def precision_recall_curve(self, ranked_list: List[int], relevant: List[int]) -> List[Dict]:
        """
        Calcule la courbe Précision-Rappel pour tous les documents
        Retourne une liste de points avec toutes les informations
        """
        precision_recall_points = []
        relevant_set = set(relevant)
        num_relevant = len(relevant)
        
        if num_relevant == 0:
            return []
        
        relevant_found = 0
        
        for rank, doc_id in enumerate(ranked_list, start=1):
            is_relevant = doc_id in relevant_set
            
            if is_relevant:
                relevant_found += 1
                recall = relevant_found / num_relevant
                precision = relevant_found / rank
                
                precision_recall_points.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'is_relevant': True,
                    'recall': recall,
                    'precision': precision,
                    'relevant_found': relevant_found
                })
            else:
                precision_recall_points.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'is_relevant': False,
                    'recall': None,
                    'precision': None,
                    'relevant_found': relevant_found
                })
        
        return precision_recall_points
    
    def interpolate_precision_recall(self, precision_recall_points: List[Dict]) -> List[Tuple[float, float]]:
        """
        Interpole la courbe Précision-Rappel aux 11 points standards [0.0, 0.1, ..., 1.0]
        P(rj) = max P(r) pour tous r ≥ rj
        """
        relevant_points = [(p['recall'], p['precision']) 
                          for p in precision_recall_points if p['is_relevant']]
        
        if not relevant_points:
            return [(r/10, 0.0) for r in range(11)]
        
        # Ajouter le point (0, max_precision) si nécessaire
        if relevant_points[0][0] > 0:
            max_precision = max([p[1] for p in relevant_points])
            relevant_points.insert(0, (0.0, max_precision))
        
        standard_recalls = [r/10 for r in range(11)]
        interpolated_curve = []
        
        for rj in standard_recalls:
            precisions_at_or_above = [p for r, p in relevant_points if r >= rj]
            
            if precisions_at_or_above:
                max_precision = max(precisions_at_or_above)
                interpolated_curve.append((rj, max_precision))
            else:
                interpolated_curve.append((rj, 0.0))
        
        return interpolated_curve
    
    # ==================== AVERAGE PRECISION ====================
    
    def average_precision(self, ranked_list: List[int], relevant: List[int]) -> float:
        """
        AP = (1/R) * Σ P(k) pour chaque position k où un document pertinent apparaît
        """
        relevant_set = set(relevant)
        num_relevant = len(relevant)
        
        if num_relevant == 0:
            return 0.0
        
        relevant_retrieved = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(ranked_list):
            if doc_id in relevant_set:
                relevant_retrieved += 1
                precision_at_k = relevant_retrieved / (i + 1)
                sum_precisions += precision_at_k
        
        return sum_precisions / num_relevant
    
    def interpolated_average_precision(self, precision_recall_points: List[Dict]) -> float:
        """
        Calcule l'AP en utilisant les précisions interpolées aux 11 points standards
        """
        interpolated_curve = self.interpolate_precision_recall(precision_recall_points)
        precisions = [p[1] for p in interpolated_curve]
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    # ==================== PRÉCISION@K ====================
    
    def precision_at_k(self, ranked_list: List[int], relevant: List[int], k: int) -> float:
        """
        P@K = Nombre de documents pertinents dans les K premiers / K
        """
        if k == 0 or k > len(ranked_list):
            k = len(ranked_list)
        
        top_k = ranked_list[:k]
        relevant_set = set(relevant)
        relevant_in_top_k = len(set(top_k) & relevant_set)
        
        return relevant_in_top_k / k
    
    # ==================== R-PRECISION ====================
    
    def r_precision(self, ranked_list: List[int], relevant: List[int]) -> float:
        """
        R-Precision: Précision aux R premiers documents (R = nombre de docs pertinents)
        """
        r = len(relevant)
        return self.precision_at_k(ranked_list, relevant, r)
    
    # ==================== RECIPROCAL RANK ====================
    
    def reciprocal_rank(self, ranked_list: List[int], relevant: List[int]) -> float:
        """
        RR = 1 / rang du premier document pertinent
        """
        relevant_set = set(relevant)
        
        for i, doc_id in enumerate(ranked_list):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    # ==================== DCG & nDCG ====================
    
    def dcg_at_k(self, ranked_list: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        """
        DCG@K = rel_1 + Σ (rel_i / log2(i+1)) pour i de 2 à k
        """
        dcg = 0.0
        
        for i, doc_id in enumerate(ranked_list[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            
            if i == 0:
                dcg += rel
            else:
                dcg += rel / math.log2(i + 1)
        
        return dcg
    
    def idcg_at_k(self, relevance_scores: Dict[int, float], k: int) -> float:
        """
        IDCG@K: DCG idéal (documents triés par pertinence décroissante)
        """
        sorted_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        ideal_list = [doc_id for doc_id, score in sorted_docs]
        
        return self.dcg_at_k(ideal_list, relevance_scores, k)
    
    def ndcg_at_k(self, ranked_list: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        """
        nDCG@K = DCG@K / IDCG@K
        """
        dcg = self.dcg_at_k(ranked_list, relevance_scores, k)
        idcg = self.idcg_at_k(relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    # ==================== VISUALISATION ====================
    
    def plot_precision_recall_curve(self, precision_recall_points: List[Dict], 
                                    query_id: int, interpolated: bool = False):
        """Trace la courbe Précision-Rappel pour une requête"""
        if interpolated:
            curve = self.interpolate_precision_recall(precision_recall_points)
            recalls = [p[0] for p in curve]
            precisions = [p[1] for p in curve]
            title = f"Courbe Précision-Rappel Interpolée - {self.model_name} - Requête {query_id}"
            filename = f"query_{query_id}_interpolated.png"
            save_dir = self.dirs['curves_interpolated']
        else:
            relevant_points = [p for p in precision_recall_points if p['is_relevant']]
            recalls = [p['recall'] for p in relevant_points]
            precisions = [p['precision'] for p in relevant_points]
            title = f"Courbe Précision-Rappel - {self.model_name} - Requête {query_id}"
            filename = f"query_{query_id}_non_interpolated.png"
            save_dir = self.dirs['curves_non_interpolated']
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Rappel', fontsize=12)
        plt.ylabel('Précision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_all_queries_curves(self, all_results: Dict, interpolated: bool = False):
        """Trace toutes les courbes sur un même graphique"""
        plt.figure(figsize=(14, 8))
        
        for query_id, result in sorted(all_results.items()):
            if interpolated:
                curve = self.interpolate_precision_recall(result['precision_recall_curve'])
                recalls = [p[0] for p in curve]
                precisions = [p[1] for p in curve]
            else:
                relevant_points = [p for p in result['precision_recall_curve'] if p['is_relevant']]
                recalls = [p['recall'] for p in relevant_points]
                precisions = [p['precision'] for p in relevant_points]
            
            plt.plot(recalls, precisions, '-o', linewidth=1.5, markersize=4, alpha=0.7, label=f"Q{query_id}")
        
        plt.xlabel('Rappel', fontsize=12)
        plt.ylabel('Précision', fontsize=12)
        
        if interpolated:
            title = f"Courbes Précision-Rappel Interpolées - {self.model_name} - Toutes les Requêtes"
            filename = "all_queries_interpolated.png"
        else:
            title = f"Courbes Précision-Rappel - {self.model_name} - Toutes les Requêtes"
            filename = "all_queries_non_interpolated.png"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        
        save_path = os.path.join(self.dirs['all_queries'], filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    # ==================== ÉVALUATION COMPLÈTE ====================
    
    def evaluate_query(self, query_id: int, ranked_list: List[int], 
                      relevance_scores: Dict[int, float] = None,
                      verbose: bool = False) -> Dict:
        """
        Évalue une requête avec toutes les métriques
        """
        if query_id not in self.relevance_judgments:
            return None
        
        relevant = self.relevance_judgments[query_id]
        total_relevant = len(relevant)
        
        # Courbe Précision-Rappel complète
        precision_recall_curve = self.precision_recall_curve(ranked_list, relevant)
        
        # Compter les documents pertinents trouvés
        relevant_found = sum(1 for p in precision_recall_curve if p['is_relevant'])
        
        # Métriques de base
        precision_final = relevant_found / len(ranked_list) if len(ranked_list) > 0 else 0
        recall_final = relevant_found / total_relevant if total_relevant > 0 else 0
        f_measure = self.f1_score(precision_final, recall_final)
        
        # Average Precision
        avg_precision = self.average_precision(ranked_list, relevant)
        interpolated_curve = self.interpolate_precision_recall(precision_recall_curve)
        avg_precision_interpolated = self.interpolated_average_precision(precision_recall_curve)
        
        # P@5 et P@10
        p_at_5 = self.precision_at_k(ranked_list, relevant, 5)
        p_at_10 = self.precision_at_k(ranked_list, relevant, 10)
        
        # R-Precision
        r_precision = self.r_precision(ranked_list, relevant)
        
        # Reciprocal Rank
        reciprocal_rank = self.reciprocal_rank(ranked_list, relevant)
        
        result = {
            'query_id': query_id,
            'ranking': ranked_list,
            'relevant_docs': relevant,
            'precision_recall_curve': precision_recall_curve,
            'interpolated_curve': interpolated_curve,
            'total_relevant': total_relevant,
            'relevant_found': relevant_found,
            'precision_final': precision_final,
            'recall_final': recall_final,
            'f_measure': f_measure,
            'average_precision': avg_precision,
            'average_precision_interpolated': avg_precision_interpolated,
            'p_at_5': p_at_5,
            'p_at_10': p_at_10,
            'r_precision': r_precision,
            'reciprocal_rank': reciprocal_rank
        }
        
        # DCG et nDCG si les scores sont fournis
        if relevance_scores:
            dcg_at_20 = self.dcg_at_k(ranked_list, relevance_scores, 20)
            ndcg_at_20 = self.ndcg_at_k(ranked_list, relevance_scores, 20)
            result['dcg_at_20'] = dcg_at_20
            result['ndcg_at_20'] = ndcg_at_20
        
        if verbose:
            self._print_query_results(result)
        
        return result
    
    def _print_query_results(self, result: Dict):
        """Affiche les résultats d'une requête"""
        print(f"\n{'='*120}")
        print(f"📊 REQUÊTE {result['query_id']} - {self.model_name}")
        print(f"{'='*120}")
        print(f"Docs pertinents (R): {result['total_relevant']}")
        print(f"Docs pertinents trouvés: {result['relevant_found']}")
        print(f"{'='*120}\n")
        
        print(f"{'Rang':<6} {'Doc':<8} {'Pert.':<8} {'Rappel':<12} {'Précision':<12}")
        print(f"{'-'*120}")
        
        for point in result['precision_recall_curve']:
            rank = point['rank']
            doc_id = point['doc_id']
            is_rel = point['is_relevant']
            recall = f"{point['recall']:.4f}" if is_rel else "-"
            precision = f"{point['precision']:.4f}" if is_rel else "-"
            pert_mark = "✓" if is_rel else "✗"
            
            print(f"{rank:<6} {doc_id:<8} {pert_mark:<8} {recall:<12} {precision:<12}")
        
        print(f"\n{'='*120}")
        print(f"✅ MÉTRIQUES:")
        print(f"   - Rappel: {result['recall_final']:.4f}")
        print(f"   - Précision: {result['precision_final']:.4f}")
        print(f"   - F-Mesure: {result['f_measure']:.4f}")
        print(f"   - P@5: {result['p_at_5']:.4f}")
        print(f"   - P@10: {result['p_at_10']:.4f}")
        print(f"   - R-Precision: {result['r_precision']:.4f}")
        print(f"   - Reciprocal Rank: {result['reciprocal_rank']:.4f}")
        print(f"   - Average Precision: {result['average_precision']:.4f}")
        print(f"   - AP Interpolée: {result['average_precision_interpolated']:.4f}")
        
        if 'dcg_at_20' in result:
            print(f"   - DCG@20: {result['dcg_at_20']:.4f}")
            print(f"   - nDCG@20: {result['ndcg_at_20']:.4f}")
        
        print(f"{'='*120}\n")
    
    def evaluate_all_queries(self, results_per_query: Dict[int, List[int]],
                            relevance_scores_per_query: Dict[int, Dict[int, float]] = None,
                            verbose: bool = False,
                            plot_curves: bool = True,
                            save_results: bool = True) -> Dict:
        """
        Évalue toutes les requêtes
        
        Args:
            results_per_query: Dict {query_id: [liste ordonnée des doc_ids]}
            relevance_scores_per_query: Dict {query_id: {doc_id: score}} pour DCG/nDCG
            verbose: Afficher les détails
            plot_curves: Générer les courbes
            save_results: Sauvegarder les résultats dans un fichier
        """
        all_results = {}
        
        print("\n" + "="*120)
        print(f"🚀 ÉVALUATION {self.model_name} - TOUTES LES REQUÊTES")
        print("="*120)
        
        for query_id, ranked_list in results_per_query.items():
            if query_id not in self.relevance_judgments:
                continue
            
            rel_scores = relevance_scores_per_query.get(query_id) if relevance_scores_per_query else None
            result = self.evaluate_query(query_id, ranked_list, rel_scores, verbose=verbose)
            
            if result:
                all_results[query_id] = result
                
                if plot_curves:
                    self.plot_precision_recall_curve(result['precision_recall_curve'], query_id, False)
                    self.plot_precision_recall_curve(result['precision_recall_curve'], query_id, True)
                
                print(f"✅ Q{query_id:2d} | AP: {result['average_precision']:.4f} | "
                      f"P@5: {result['p_at_5']:.4f} | P@10: {result['p_at_10']:.4f} | "
                      f"RR: {result['reciprocal_rank']:.4f}")
        
        # Courbes globales
        if plot_curves:
            print("\n📊 Génération des courbes globales...")
            self.plot_all_queries_curves(all_results, False)
            self.plot_all_queries_curves(all_results, True)
        
        # Calculer les moyennes
        map_score = sum(r['average_precision'] for r in all_results.values()) / len(all_results)
        mean_p5 = sum(r['p_at_5'] for r in all_results.values()) / len(all_results)
        mean_p10 = sum(r['p_at_10'] for r in all_results.values()) / len(all_results)
        mean_r_prec = sum(r['r_precision'] for r in all_results.values()) / len(all_results)
        mean_rr = sum(r['reciprocal_rank'] for r in all_results.values()) / len(all_results)
        
        print("\n" + "="*120)
        print("📈 MÉTRIQUES GLOBALES:")
        print(f"   - MAP: {map_score:.4f}")
        print(f"   - Mean P@5: {mean_p5:.4f}")
        print(f"   - Mean P@10: {mean_p10:.4f}")
        print(f"   - Mean R-Precision: {mean_r_prec:.4f}")
        print(f"   - MRR: {mean_rr:.4f}")
        
        if relevance_scores_per_query:
            mean_dcg = sum(r['dcg_at_20'] for r in all_results.values()) / len(all_results)
            mean_ndcg = sum(r['ndcg_at_20'] for r in all_results.values()) / len(all_results)
            print(f"   - Mean DCG@20: {mean_dcg:.4f}")
            print(f"   - Mean nDCG@20: {mean_ndcg:.4f}")
        
        print("="*120)
        
        if save_results:
            self.save_detailed_results(all_results)
        
        return all_results
    
    def save_detailed_results(self, all_results: Dict):
        """Sauvegarde les résultats détaillés dans un fichier texte"""
        output_file = f"results/{self.model_name}_results.txt"
        os.makedirs("results", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write(f"RÉSULTATS {self.model_name} - ÉVALUATION COMPLÈTE\n")
            f.write("="*120 + "\n\n")
            
            # Calculer MAP et MAP interpolée
            map_score = sum(r['average_precision'] for r in all_results.values()) / len(all_results) if all_results else 0.0
            map_interpolated = sum(r['average_precision_interpolated'] for r in all_results.values()) / len(all_results) if all_results else 0.0
            
            # Calculer les autres moyennes
            mean_p5 = sum(r['p_at_5'] for r in all_results.values()) / len(all_results) if all_results else 0.0
            mean_p10 = sum(r['p_at_10'] for r in all_results.values()) / len(all_results) if all_results else 0.0
            mean_r_prec = sum(r['r_precision'] for r in all_results.values()) / len(all_results) if all_results else 0.0
            mean_rr = sum(r['reciprocal_rank'] for r in all_results.values()) / len(all_results) if all_results else 0.0
            
            # Vérifier si DCG/nDCG sont disponibles
            has_dcg = 'dcg_at_20' in list(all_results.values())[0] if all_results else False
            if has_dcg:
                mean_dcg = sum(r['dcg_at_20'] for r in all_results.values()) / len(all_results)
                mean_ndcg = sum(r['ndcg_at_20'] for r in all_results.values()) / len(all_results)
            
            # Afficher les métriques globales en haut
            f.write("="*120 + "\n")
            f.write("📈 MÉTRIQUES GLOBALES MOYENNES\n")
            f.write("="*120 + "\n")
            f.write(f"MAP (Mean Average Precision):        {map_score:.4f}\n")
            f.write(f"MAP Interpolée:                      {map_interpolated:.4f}\n")
            f.write(f"Mean P@5:                            {mean_p5:.4f}\n")
            f.write(f"Mean P@10:                           {mean_p10:.4f}\n")
            f.write(f"Mean R-Precision:                    {mean_r_prec:.4f}\n")
            f.write(f"MRR (Mean Reciprocal Rank):          {mean_rr:.4f}\n")
            if has_dcg:
                f.write(f"Mean DCG@20:                         {mean_dcg:.4f}\n")
                f.write(f"Mean nDCG@20:                        {mean_ndcg:.4f}\n")
            f.write("="*120 + "\n\n")
            
            for query_id in sorted(all_results.keys()):
                result = all_results[query_id]
                
                f.write("="*120 + "\n")
                f.write(f"REQUÊTE {query_id}\n")
                f.write("="*120 + "\n")
                f.write(f"Documents pertinents (R): {result['total_relevant']}\n")
                f.write(f"Documents pertinents trouvés: {result['relevant_found']}\n\n")
                
                f.write("MÉTRIQUES:\n")
                f.write("-"*120 + "\n")
                f.write(f"Rappel:              {result['recall_final']:.4f}\n")
                f.write(f"Précision:           {result['precision_final']:.4f}\n")
                f.write(f"F-Mesure:            {result['f_measure']:.4f}\n")
                f.write(f"P@5:                 {result['p_at_5']:.4f}\n")
                f.write(f"P@10:                {result['p_at_10']:.4f}\n")
                f.write(f"R-Precision:         {result['r_precision']:.4f}\n")
                f.write(f"Reciprocal Rank:     {result['reciprocal_rank']:.4f}\n")
                f.write(f"Average Precision:   {result['average_precision']:.4f}\n")
                f.write(f"AP Interpolée:       {result['average_precision_interpolated']:.4f}\n")
                
                if 'dcg_at_20' in result:
                    f.write(f"DCG@20:              {result['dcg_at_20']:.4f}\n")
                    f.write(f"nDCG@20:             {result['ndcg_at_20']:.4f}\n")
                
                f.write("\nCLASSEMENT COMPLET:\n")
                f.write("-"*120 + "\n")
                f.write(f"{'Rang':<6} {'Doc':<8} {'Pert.':<8} {'Rappel':<12} {'Précision':<12}\n")
                f.write("-"*120 + "\n")
                
                for point in result['precision_recall_curve']:
                    rank = point['rank']
                    doc_id = point['doc_id']
                    is_rel = point['is_relevant']
                    recall = f"{point['recall']:.4f}" if is_rel else "-"
                    precision = f"{point['precision']:.4f}" if is_rel else "-"
                    pert_mark = "✓" if is_rel else "✗"
                    
                    f.write(f"{rank:<6} {doc_id:<8} {pert_mark:<8} {recall:<12} {precision:<12}\n")
                
                f.write("\n\n")
        
        print(f"\n✅ Résultats sauvegardés: {output_file}")