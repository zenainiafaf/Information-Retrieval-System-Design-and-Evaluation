

import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple
from medline_parser import MEDDocument, MEDQuery


class MEDLINEPreprocessor:
    """Classe pour le preprocessing des documents et requêtes MEDLINE"""
    
    def __init__(self):
        # Tokenizer - MÊME REGEX que lab1.py
        self.tokenizer = RegexpTokenizer(
            r'(?:[A-Za-z]\.)+|'                           # Abbreviations like D.Z.A
            r'[A-Za-z]+[\-@]\d+(?:\.\d+)?|'               # Words combined with numbers, e.g., data-1
            r'\d+(?:[\.\,\-]\d+)*%?|'                     # Numbers with decimals, separators, or percentages
            r'[A-Za-z]+'                                   # Simple words (alphabetic)
        )
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Porter Stemmer
        self.porter = PorterStemmer()
        
        # Stockage des résultats
        self.processed_docs = {}
        self.processed_queries = {}
        self.doc_term_freqs = {}
        self.query_term_freqs = {}
        self.tf_idf_weights = {}
        self.document_frequency = {}
        self.N = 0
    
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize le texte en utilisant RegexpTokenizer
        
        Args:
            text (str): Texte à tokeniser
            
        Returns:
            List[str]: Liste des tokens en minuscules
        """
        tokens = [token.lower() for token in self.tokenizer.tokenize(text)]
        return tokens
    
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Supprime les stop words de la liste de tokens
        
        Args:
            tokens (List[str]): Liste des tokens
            
        Returns:
            List[str]: Liste des tokens sans stop words
        """
        filtered = [token for token in tokens if token not in self.stop_words]
        return filtered
    
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Applique le Porter Stemmer sur les tokens
        
        Args:
            tokens (List[str]): Liste des tokens
            
        Returns:
            List[str]: Liste des tokens normalisés
        """
        stemmed = [self.porter.stem(token) for token in tokens]
        return stemmed
    
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Pipeline complet de preprocessing pour un texte
        
        Args:
            text (str): Texte à preprocesser
            
        Returns:
            List[str]: Liste des termes preprocessés
        """
        # 1. Tokenization
        tokens = self.tokenize(text)
        
        # 2. Remove stop words
        tokens = self.remove_stopwords(tokens)
        
        # 3. Stemming avec Porter
        tokens = self.stem_tokens(tokens)
        
        return tokens
    
    
    def preprocess_documents(self, documents: List[MEDDocument], verbose: bool = True):
        """
        Preprocess tous les documents MEDLINE
        
        Args:
            documents (List[MEDDocument]): Liste des documents
            verbose (bool): Afficher les informations de progression
        """
        if verbose:
            print("\n" + "=" * 80)
            print("PREPROCESSING DES DOCUMENTS MEDLINE")
            print("=" * 80)
        
        self.N = len(documents)
        
        for doc in documents:
            # Obtenir le texte complet (titre + abstract)
            full_text = doc.get_full_text()
            
            # Appliquer le preprocessing
            processed_terms = self.preprocess_text(full_text)
            
            # Stocker les résultats
            self.processed_docs[doc.doc_id] = processed_terms
            
            if verbose and doc.doc_id <= 3:
                print(f"\nDocument {doc.doc_id}:")
                print(f"  Texte original: {full_text[:100]}...")
                print(f"  Nombre de termes après preprocessing: {len(processed_terms)}")
                print(f"  Premiers termes: {processed_terms[:10]}")
        
        if verbose:
            print(f"\n✓ {len(self.processed_docs)} documents preprocessés")
            print("=" * 80)
    
    
    def preprocess_queries(self, queries: List[MEDQuery], verbose: bool = True):
        """
        Preprocess toutes les requêtes MEDLINE
        
        Args:
            queries (List[MEDQuery]): Liste des requêtes
            verbose (bool): Afficher les informations de progression
        """
        if verbose:
            print("\n" + "=" * 80)
            print("PREPROCESSING DES REQUÊTES MEDLINE")
            print("=" * 80)
        
        for query in queries:
            # Appliquer le preprocessing
            processed_terms = self.preprocess_text(query.text)
            
            # Stocker les résultats
            self.processed_queries[query.query_id] = processed_terms
            
            if verbose and query.query_id <= 3:
                print(f"\nRequête {query.query_id}:")
                print(f"  Texte original: {query.text[:100]}...")
                print(f"  Nombre de termes: {len(processed_terms)}")
                print(f"  Termes: {processed_terms}")
        
        if verbose:
            print(f"\n✓ {len(self.processed_queries)} requêtes preprocessées")
            print("=" * 80)
    
    
    def calculate_term_frequencies(self, verbose: bool = True):
        """
        Calcule les fréquences des termes pour chaque document
        
        Args:
            verbose (bool): Afficher les informations
        """
        if verbose:
            print("\n" + "=" * 80)
            print("CALCUL DES FRÉQUENCES DES TERMES")
            print("=" * 80)
        
        # Fréquences des documents - utilise nltk.FreqDist comme lab1.py
        for doc_id, terms in self.processed_docs.items():
            freq_dist = nltk.FreqDist(terms)
            self.doc_term_freqs[doc_id] = dict(freq_dist)
        
        # Fréquences des requêtes
        for query_id, terms in self.processed_queries.items():
            freq_dist = nltk.FreqDist(terms)
            self.query_term_freqs[query_id] = dict(freq_dist)
        
        if verbose:
            print(f"✓ Fréquences calculées pour {len(self.doc_term_freqs)} documents")
            print(f"✓ Fréquences calculées pour {len(self.query_term_freqs)} requêtes")
            
            # Afficher exemple
            if self.doc_term_freqs:
                first_doc_id = list(self.doc_term_freqs.keys())[0]
                print(f"\nExemple - Document {first_doc_id}:")
                sorted_freq = OrderedDict(sorted(
                    self.doc_term_freqs[first_doc_id].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
                print(f"  Top 5 termes: {dict(sorted_freq)}")
    
    
    def calculate_document_frequency(self, verbose: bool = True):
        """
        Calcule la fréquence documentaire
        
        Args:
            verbose (bool): Afficher les informations
        """
        if verbose:
            print("\n" + "=" * 80)
            print("CALCUL DE LA FRÉQUENCE DOCUMENTAIRE")
            print("=" * 80)
        
        self.document_frequency = {}
        for doc_id, terms in self.processed_docs.items():
            unique_terms = set(terms)  # Get unique terms in this document
            for term in unique_terms:
                if term not in self.document_frequency:
                    self.document_frequency[term] = 0
                self.document_frequency[term] += 1
        
        if verbose:
            print(f"✓ Nombre total de termes uniques: {len(self.document_frequency)}")
            
            # Afficher les termes les plus fréquents
            sorted_df = sorted(
                self.document_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            print(f"\nTermes apparaissant dans le plus de documents:")
            for term, freq in sorted_df:
                print(f"  {term}: {freq} documents")
    
    
    def compute_tf_idf(self, terms_per_doc: dict):
        """
        Calcule TF-IDF 
        
        Args:
            terms_per_doc: Dictionnaire {doc_id: [liste de termes]}
            
        Returns:
            tuple: (doc_term_freqs, tf_idf_weights, document_frequency, N)
        """
        N = len(terms_per_doc)

        # Document frequency n_t
        document_frequency: dict[str, int] = defaultdict(int)
        for terms in terms_per_doc.values():
            for term in set(terms):
                document_frequency[term] += 1

        # Term frequencies per document and max freq per doc
        doc_term_freqs: dict[str, dict[str, int]] = {}
        max_freq_per_doc: dict[str, int] = {}
        for doc_id, terms in terms_per_doc.items():
            tf: dict[str, int] = defaultdict(int)
            for t in terms:
                tf[t] += 1
            doc_term_freqs[doc_id] = dict(tf)
            max_freq_per_doc[doc_id] = max(tf.values()) if tf else 1

        # TF-IDF weights per document
        tf_idf_weights: dict[str, dict[str, float]] = {}
        for doc_id, tf in doc_term_freqs.items():
            tf_idf_weights[doc_id] = {}
            maxf = max_freq_per_doc[doc_id]
            for term, freq in tf.items():
                tf_norm = freq / maxf
                # MÊME FORMULE IDF que lab1.py
                idf = math.log10(N / (document_frequency[term]) + 1)
                tf_idf_weights[doc_id][term] = tf_norm * idf

        return doc_term_freqs, tf_idf_weights, document_frequency, N
    
    
    def calculate_tf_idf(self, verbose: bool = True):
        """
        Calcule les poids TF-IDF en utilisant la fonction compute_tf_idf
        
        Args:
            verbose (bool): Afficher les informations
        """
        if verbose:
            print("\n" + "=" * 80)
            print("CALCUL DES POIDS TF-IDF")
            print("=" * 80)
        
        self.doc_term_freqs, self.tf_idf_weights, self.document_frequency, self.N = \
            self.compute_tf_idf(self.processed_docs)
        
        if verbose:
            print(f"✓ Poids TF-IDF calculés pour {len(self.tf_idf_weights)} documents")
            print(f"✓ Total de termes uniques: {len(self.document_frequency)}")
            
            # Afficher exemple
            if self.tf_idf_weights:
                first_doc_id = list(self.tf_idf_weights.keys())[0]
                sorted_weights = sorted(
                    self.tf_idf_weights[first_doc_id].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                print(f"\nExemple - Document {first_doc_id} - Top 5 termes par TF-IDF:")
                for term, weight in sorted_weights:
                    print(f"  {term}: {weight:.6f}")
    
    
    def save_document_term_matrix(self, output_path: str, verbose: bool = True):
        """
        Sauvegarde la Document-Term Matrix: <Document> <Term> <Frequency>
        
        Args:
            output_path (str): Chemin du fichier de sortie
            verbose (bool): Afficher les informations
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for doc_id in sorted(self.doc_term_freqs.keys()):
                # Sort terms alphabetically for each document
                for term in sorted(self.doc_term_freqs[doc_id].keys()):
                    freq = self.doc_term_freqs[doc_id][term]
                    f.write(f"{doc_id} {term} {freq}\n")
        
        if verbose:
            print(f"\n✓ Document-Term Matrix créée: {output_path}")
            
            # Afficher quelques lignes
            print("\n  Aperçu du fichier:")
            with open(output_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"    {line.strip()}")
                    else:
                        break
    
    
    def save_inverted_index(self, output_path: str, verbose: bool = True):
        """
        Sauvegarde l'Inverted Index: <Term> <Document> <Frequency> <Weight>
       
        
        Args:
            output_path (str): Chemin du fichier de sortie
            verbose (bool): Afficher les informations
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build inverted index structure 
        inverted_index = defaultdict(list)
        for doc_id in self.doc_term_freqs.keys():
            for term in self.doc_term_freqs[doc_id].keys():
                freq = self.doc_term_freqs[doc_id][term]
                weight = self.tf_idf_weights[doc_id][term]
                inverted_index[term].append((doc_id, freq, weight))
        
        # Write inverted index file (sorted by term)
        with open(output_path, "w", encoding="utf-8") as f:
            for term in sorted(inverted_index.keys()):
                for doc_id, freq, weight in sorted(inverted_index[term]):
                    f.write(f"{term} {doc_id} {freq} {weight:.6f}\n")
        
        if verbose:
            print(f"\n✓ Inverted Index créé: {output_path}")
            
            # Afficher quelques lignes
            print("\n  Aperçu du fichier:")
            with open(output_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"    {line.strip()}")
                    else:
                        break
    
    
    def get_statistics(self):
        """Affiche les statistiques du preprocessing"""
        print("\n" + "=" * 80)
        print("📊 STATISTIQUES DU PREPROCESSING")
        print("=" * 80)
        
        print(f"\n📄 Documents:")
        print(f"   - Nombre total: {len(self.processed_docs)}")
        if self.processed_docs:
            avg_terms = sum(len(terms) for terms in self.processed_docs.values()) / len(self.processed_docs)
            print(f"   - Moyenne de termes par document: {avg_terms:.1f}")
        
        print(f"\n🔍 Requêtes:")
        print(f"   - Nombre total: {len(self.processed_queries)}")
        if self.processed_queries:
            avg_terms = sum(len(terms) for terms in self.processed_queries.values()) / len(self.processed_queries)
            print(f"   - Moyenne de termes par requête: {avg_terms:.1f}")
        
        print(f"\n📚 Vocabulaire:")
        print(f"   - Nombre de termes uniques: {len(self.document_frequency)}")
        
        print("=" * 80 + "\n")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du preprocessor
    """
    from medline_parser import parse_med_all, parse_med_qry
    
    print("🚀 MEDLINE PREPROCESSING - Exemple d'utilisation\n")
    
    # Chemins 
    MED_ALL_PATH = r"data\MED.ALL"
    MED_QRY_PATH = r"data\MED.QRY"
    OUTPUT_DIR = r"output"
    
    # Parser les données
    print("📄 Parsing des données MEDLINE...")
    documents = parse_med_all(MED_ALL_PATH)
    queries = parse_med_qry(MED_QRY_PATH)
    
    # Créer le preprocessor
    preprocessor = MEDLINEPreprocessor()
    
    # Preprocessing des documents
    preprocessor.preprocess_documents(documents, verbose=True)
    
    # Preprocessing des requêtes
    preprocessor.preprocess_queries(queries, verbose=True)
    
    # Calculs - MÊME ORDRE que lab1.py
    preprocessor.calculate_term_frequencies(verbose=True)
    preprocessor.calculate_document_frequency(verbose=True)
    preprocessor.calculate_tf_idf(verbose=True)
    
    # Build: Document-Term Matrix et Inverted Index (LAB 5 requirements)
    print("\n" + "=" * 80)
    print("BUILD INDEXES WITH FREQUENCY AND WEIGHT")
    print("=" * 80)
    
    print("\n1) Creating Document-Term Matrix...")
    preprocessor.save_document_term_matrix(
        os.path.join(OUTPUT_DIR, "document_term_matrix.txt"),
        verbose=True
    )
    
    print("\n2) Creating Inverted Index...")
    preprocessor.save_inverted_index(
        os.path.join(OUTPUT_DIR, "inverted_index.txt"),
        verbose=True
    )
    
    # Statistiques
    preprocessor.get_statistics()
    
    print("✅ Preprocessing terminé avec succès!")
