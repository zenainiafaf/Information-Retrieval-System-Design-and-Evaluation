

import re
from typing import List, Dict, Tuple


class MEDDocument:
    """Classe représentant un document médical"""
    def __init__(self, doc_id: int, title: str = "", abstract: str = "", 
                 authors: str = "", source: str = ""):
        self.doc_id = doc_id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.source = source
        
    def get_full_text(self) -> str:
        """Retourne le texte complet (titre + résumé)"""
        return f"{self.title} {self.abstract}".strip()
    
    def __repr__(self):
        return f"MEDDocument(id={self.doc_id}, title='{self.title[:50]}...')"


class MEDQuery:
    """Classe représentant une requête"""
    def __init__(self, query_id: int, text: str = ""):
        self.query_id = query_id
        self.text = text
    
    def __repr__(self):
        return f"MEDQuery(id={self.query_id}, text='{self.text[:50]}...')"


def parse_med_all(filepath: str) -> List[MEDDocument]:
    """
    Parse le fichier MED.ALL contenant les documents médicaux.
    
    Format du fichier :
    .I [numéro du document]
    .T [titre]
    .W [résumé/abstract]
    .A [auteurs]
    .S [source]
    
    Args:
        filepath (str): Chemin vers le fichier MED.ALL
        
    Returns:
        List[MEDDocument]: Liste des documents parsés
    """
    documents = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {filepath} n'existe pas!")
        return documents
    
    # Séparer les documents (chaque document commence par .I)
    doc_sections = re.split(r'\.I\s+(\d+)', content)[1:]  # [1:] pour ignorer le premier élément vide
    
    # Traiter par paires (id, contenu)
    for i in range(0, len(doc_sections), 2):
        if i + 1 >= len(doc_sections):
            break
            
        doc_id = int(doc_sections[i].strip())
        doc_content = doc_sections[i + 1]
        
        # Extraire les différentes sections
        title = ""
        abstract = ""
        authors = ""
        source = ""
        
        # Extraire le titre (.T)
        title_match = re.search(r'\.T\s+(.*?)(?=\.[AWSIB]|\Z)', doc_content, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
        
        # Extraire l'abstract (.W)
        abstract_match = re.search(r'\.W\s+(.*?)(?=\.[ATSIB]|\Z)', doc_content, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        
        # Extraire les auteurs (.A)
        authors_match = re.search(r'\.A\s+(.*?)(?=\.[WTSIB]|\Z)', doc_content, re.DOTALL)
        if authors_match:
            authors = authors_match.group(1).strip()
        
        # Extraire la source (.S)
        source_match = re.search(r'\.S\s+(.*?)(?=\.[WTAIB]|\Z)', doc_content, re.DOTALL)
        if source_match:
            source = source_match.group(1).strip()
        
        # Créer l'objet document
        doc = MEDDocument(
            doc_id=doc_id,
            title=title,
            abstract=abstract,
            authors=authors,
            source=source
        )
        
        documents.append(doc)
    
    print(f"✅ {len(documents)} documents parsés depuis {filepath}")
    return documents


def parse_med_qry(filepath: str) -> List[MEDQuery]:
    """
    Parse le fichier MED.QRY contenant les requêtes.
    
    Format du fichier :
    .I [numéro de la requête]
    .W [texte de la requête]
    
    Args:
        filepath (str): Chemin vers le fichier MED.QRY
        
    Returns:
        List[MEDQuery]: Liste des requêtes parsées
    """
    queries = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {filepath} n'existe pas!")
        return queries
    
    # Séparer les requêtes (chaque requête commence par .I)
    query_sections = re.split(r'\.I\s+(\d+)', content)[1:]
    
    # Traiter par paires (id, contenu)
    for i in range(0, len(query_sections), 2):
        if i + 1 >= len(query_sections):
            break
            
        query_id = int(query_sections[i].strip())
        query_content = query_sections[i + 1]
        
        # Extraire le texte de la requête (.W)
        text = ""
        text_match = re.search(r'\.W\s+(.*?)(?=\.I|\Z)', query_content, re.DOTALL)
        if text_match:
            text = text_match.group(1).strip()
        
        # Créer l'objet requête
        query = MEDQuery(query_id=query_id, text=text)
        queries.append(query)
    
    print(f"✅ {len(queries)} requêtes parsées depuis {filepath}")
    return queries


def parse_med_rel(filepath: str) -> Dict[int, List[int]]:
    """
    Parse le fichier MED.REL contenant les jugements de pertinence.
    
    Format du fichier :
    [query_id] 0 [doc_id] [relevance_score]
    
    Exemple :
    1 0 13 1
    1 0 14 2
    
    Args:
        filepath (str): Chemin vers le fichier MED.REL
        
    Returns:
        Dict[int, List[int]]: Dictionnaire {query_id: [liste des doc_ids pertinents]}
    """
    relevance_judgments = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {filepath} n'existe pas!")
        return relevance_judgments
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Format: query_id 0 doc_id relevance_score
        parts = line.split()
        if len(parts) >= 3:
            query_id = int(parts[0])
            doc_id = int(parts[2])
            # relevance_score = int(parts[3]) if len(parts) >= 4 else 1
            
            # Ajouter le document pertinent à la requête
            if query_id not in relevance_judgments:
                relevance_judgments[query_id] = []
            
            relevance_judgments[query_id].append(doc_id)
    
    print(f"✅ Jugements de pertinence parsés pour {len(relevance_judgments)} requêtes depuis {filepath}")
    return relevance_judgments


def parse_med_rel_with_scores(filepath: str) -> Dict[int, Dict[int, int]]:
    """
    Parse le fichier MED.REL en conservant les scores de pertinence.
    
    Format du fichier :
    [query_id] 0 [doc_id] [relevance_score]
    
    Args:
        filepath (str): Chemin vers le fichier MED.REL
        
    Returns:
        Dict[int, Dict[int, int]]: {query_id: {doc_id: relevance_score}}
    """
    relevance_judgments = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {filepath} n'existe pas!")
        return relevance_judgments
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 4:
            query_id = int(parts[0])
            doc_id = int(parts[2])
            relevance_score = int(parts[3])
            
            if query_id not in relevance_judgments:
                relevance_judgments[query_id] = {}
            
            relevance_judgments[query_id][doc_id] = relevance_score
    
    print(f"✅ Jugements de pertinence avec scores parsés pour {len(relevance_judgments)} requêtes")
    return relevance_judgments


def get_statistics(documents: List[MEDDocument], 
                   queries: List[MEDQuery],
                   relevance_judgments: Dict[int, List[int]]) -> None:
    """
    Affiche des statistiques sur le dataset MEDLINE.
    
    Args:
        documents: Liste des documents
        queries: Liste des requêtes
        relevance_judgments: Dictionnaire des jugements de pertinence
    """
    print("\n" + "="*60)
    print("📊 STATISTIQUES DU DATASET MEDLINE")
    print("="*60)
    
    print(f"\n📄 Documents:")
    print(f"   - Nombre total: {len(documents)}")
    if documents:
        avg_title_len = sum(len(doc.title.split()) for doc in documents) / len(documents)
        avg_abstract_len = sum(len(doc.abstract.split()) for doc in documents) / len(documents)
        print(f"   - Longueur moyenne du titre: {avg_title_len:.1f} mots")
        print(f"   - Longueur moyenne du résumé: {avg_abstract_len:.1f} mots")
    
    print(f"\n🔍 Requêtes:")
    print(f"   - Nombre total: {len(queries)}")
    if queries:
        avg_query_len = sum(len(q.text.split()) for q in queries) / len(queries)
        print(f"   - Longueur moyenne: {avg_query_len:.1f} mots")
    
    print(f"\n✅ Jugements de pertinence:")
    print(f"   - Nombre de requêtes avec jugements: {len(relevance_judgments)}")
    if relevance_judgments:
        total_relevant = sum(len(docs) for docs in relevance_judgments.values())
        avg_relevant = total_relevant / len(relevance_judgments)
        print(f"   - Total de jugements: {total_relevant}")
        print(f"   - Moyenne de docs pertinents par requête: {avg_relevant:.1f}")
        
        # Min et max
        min_relevant = min(len(docs) for docs in relevance_judgments.values())
        max_relevant = max(len(docs) for docs in relevance_judgments.values())
        print(f"   - Min de docs pertinents: {min_relevant}")
        print(f"   - Max de docs pertinents: {max_relevant}")
    
    print("="*60 + "\n")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation des parsers
    """
    
    print("🚀 PARSER MEDLINE - Exemple d'utilisation\n")
    
   
    MED_ALL_PATH = r"C:\Users\pc\Desktop\RI_Project\data\MED.ALL"
    MED_QRY_PATH = r"C:\Users\pc\Desktop\RI_Project\data\MED.QRY"
    MED_REL_PATH = r"C:\Users\pc\Desktop\RI_Project\data\MED.REL"
    
    # Parser les documents
    print("📄 Parsing des documents...")
    documents = parse_med_all(MED_ALL_PATH)
    
    # Parser les requêtes
    print("\n🔍 Parsing des requêtes...")
    queries = parse_med_qry(MED_QRY_PATH)
    
    # Parser les jugements de pertinence
    print("\n✅ Parsing des jugements de pertinence...")
    relevance_judgments = parse_med_rel(MED_REL_PATH)
    
    # Alternative: parser avec scores de pertinence
    relevance_with_scores = parse_med_rel_with_scores(MED_REL_PATH)
    
    # Afficher les statistiques
    get_statistics(documents, queries, relevance_judgments)
    
    # Exemples d'accès aux données
    if documents:
        print("📌 Exemple de document:")
        doc = documents[0]
        print(f"   ID: {doc.doc_id}")
        print(f"   Titre: {doc.title[:100]}...")
        print(f"   Texte complet: {doc.get_full_text()[:150]}...")
        print()
    
    if queries:
        print("📌 Exemple de requête:")
        query = queries[0]
        print(f"   ID: {query.query_id}")
        print(f"   Texte: {query.text[:150]}...")
        print()
    
    if relevance_judgments:
        print("📌 Exemple de jugement de pertinence:")
        first_query_id = list(relevance_judgments.keys())[0]
        print(f"   Requête {first_query_id} a {len(relevance_judgments[first_query_id])} documents pertinents")
        print(f"   Documents pertinents: {relevance_judgments[first_query_id][:10]}...")
        print()
    
    print("✅ Parsing terminé avec succès!")
    
   

