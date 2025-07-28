import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class ConfigurablePersonaAnalyzer:
    def __init__(self, keyword_limit: int = 10, yake_top: int = 20, yake_dedup: float = 0.3):
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en", n=1, dedupLim=yake_dedup, top=yake_top, features=None
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.keyword_limit = keyword_limit
    
    def extract_keywords(self, text: str) -> List[str]:
        try:
            keywords = self.keyword_extractor.extract_keywords(text)
            return [kw[0] for kw in keywords[:self.keyword_limit]]
        except:
            tokens = word_tokenize(text.lower())
            return [token for token in tokens if token.isalpha() and token not in self.stop_words][:self.keyword_limit]
    
    def create_interest_profile(self, persona: str, job: str) -> str:
        persona_keywords = self.extract_keywords(persona)
        job_keywords = self.extract_keywords(job)
        return f"{persona} {job} {' '.join(persona_keywords + job_keywords)}"

class AdaptiveSectionExtractor:
    def __init__(self, 
                 max_heading_length: int = 200,
                 max_heading_words: int = 10,
                 min_paragraph_length: int = 30,
                 custom_prefixes: Optional[List[str]] = None):
        self.stop_words = set(stopwords.words('english'))
        self.max_heading_length = max_heading_length
        self.max_heading_words = max_heading_words
        self.min_paragraph_length = min_paragraph_length
        self.custom_prefixes = custom_prefixes or []
    
    def extract_sections(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        sections = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: b[1])  # sort by vertical position
            
            current_section = {"title": "Untitled", "text": "", "page": page_num + 1}
            
            for block in blocks:
                block_text = block[4].strip()
                if not block_text:
                    continue
                
                # Adaptive heading detection
                if self._is_heading(block_text):
                    # Save previous section if it has content
                    if current_section["text"].strip():
                        sections.append({
                            'document': os.path.basename(pdf_path),
                            'page': current_section["page"],
                            'title': current_section["title"],
                            'text': current_section["text"].strip()
                        })
                    # Start new section
                    current_section = {"title": block_text, "text": "", "page": page_num + 1}
                else:
                    # Use first line as title if section is Untitled and text is long
                    if current_section["title"] == "Untitled":
                        lines = block_text.split('\n')
                        if len(lines) > 1 and len(lines[0].strip()) > 10:
                            current_section["title"] = lines[0].strip()
                    current_section["text"] += block_text + " "
            
            # Save last section on page
            if current_section["text"].strip():
                sections.append({
                    'document': os.path.basename(pdf_path),
                    'page': current_section["page"],
                    'title': current_section["title"],
                    'text': current_section["text"].strip()
                })
        
        # If no sections found, create sections from paragraphs
        if not sections:
            sections = self._create_sections_from_paragraphs(doc, pdf_path)
        
        return sections
    
    def _is_heading(self, text: str) -> bool:
        text_clean = text.strip()
        
        # Check for numbered headings (1.1, 1.2, etc.)
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text_clean):
            return True
        
        # Check for all-caps headings
        if len(text_clean) < self.max_heading_length and text_clean.isupper():
            return True
        
        # Check for title case headings
        if (len(text_clean) < self.max_heading_length and 
            text_clean.istitle() and 
            len(text_clean.split()) <= self.max_heading_words):
            return True
        
        # Check for custom prefixes
        text_lower = text_clean.lower()
        if any(text_lower.startswith(prefix) for prefix in self.custom_prefixes):
            return True
        
        # Check for short phrases that start with capital letters
        if len(text_clean.split()) <= self.max_heading_words and text_clean[0].isupper():
            return True
        
        return False
    
    def _create_sections_from_paragraphs(self, doc, pdf_path: str) -> List[Dict]:
        """Create sections from paragraphs when no clear headings are found"""
        sections = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > self.min_paragraph_length]
            
            for i, para in enumerate(paragraphs):
                # Use first sentence as title
                sentences = sent_tokenize(para)
                title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
                
                sections.append({
                    'document': os.path.basename(pdf_path),
                    'page': page_num + 1,
                    'title': title,
                    'text': para
                })
        
        return sections

class ConfigurableRelevanceScorer:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 tfidf_max_features: int = 2000,
                 semantic_weight: float = 0.6,
                 tfidf_weight: float = 0.4,
                 batch_size: int = 16):
        # NEW, CORRECTED LINE
        self.semantic_model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.semantic_weight = semantic_weight
        self.tfidf_weight = tfidf_weight
        self.batch_size = batch_size
    
    def compute_scores(self, interest_profile: str, section_texts: List[str]) -> np.ndarray:
        # TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([interest_profile] + section_texts)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        # Semantic
        profile_vector = self.semantic_model.encode(interest_profile)
        section_vectors = self.semantic_model.encode(section_texts, batch_size=self.batch_size, show_progress_bar=False)
        semantic_scores = cosine_similarity([profile_vector], section_vectors)[0]
        
        # Hybrid with configurable weights
        hybrid_scores = self.semantic_weight * semantic_scores + self.tfidf_weight * tfidf_scores
        return hybrid_scores
    
    def compute_subsection_scores(self, interest_profile: str, sub_texts: List[str]) -> np.ndarray:
        if not sub_texts:
            return np.array([])
        profile_vector = self.semantic_model.encode(interest_profile)
        sub_vectors = self.semantic_model.encode(sub_texts, batch_size=self.batch_size, show_progress_bar=False)
        return cosine_similarity([profile_vector], sub_vectors)[0]

def ensure_document_diversity(sections: List[Dict], top_k: int = 5) -> List[Dict]:
    """Ensure diversity by including sections from different documents"""
    if len(sections) <= top_k:
        return sections[:top_k]
    
    # Group sections by document
    doc_groups = {}
    for section in sections:
        doc = section['document']
        if doc not in doc_groups:
            doc_groups[doc] = []
        doc_groups[doc].append(section)
    
    # Sort each document's sections by score
    for doc in doc_groups:
        doc_groups[doc].sort(key=lambda x: x['score'], reverse=True)
    
    # Select top sections ensuring diversity
    diverse_sections = []
    
    # First pass: take top section from each document
    for doc, doc_sections in doc_groups.items():
        if doc_sections and len(diverse_sections) < top_k:
            diverse_sections.append(doc_sections[0])
    
    # Second pass: fill remaining slots with highest scoring sections
    remaining_sections = []
    for doc, doc_sections in doc_groups.items():
        remaining_sections.extend(doc_sections[1:])  # Skip the one we already took
    
    remaining_sections.sort(key=lambda x: x['score'], reverse=True)
    
    for section in remaining_sections:
        if len(diverse_sections) < top_k:
            diverse_sections.append(section)
        else:
            break
    
    return diverse_sections

def main(input_dir: str, 
         output_path: str, 
         persona: str, 
         job: str, 
         top_k_sections: int = 5,
         keyword_limit: int = 10,
         yake_top: int = 20,
         yake_dedup: float = 0.3,
         max_heading_length: int = 200,
         max_heading_words: int = 10,
         min_paragraph_length: int = 30,
         custom_prefixes: Optional[List[str]] = None,
         model_name: str = "all-MiniLM-L6-v2",
         tfidf_max_features: int = 2000,
         semantic_weight: float = 0.6,
         tfidf_weight: float = 0.4,
         batch_size: int = 16):
    
    print(f"🚀 General PDF Pipeline: {input_dir} | Persona: {persona} | Job: {job}")
    print(f"⚙️  Configuration:")
    print(f"   - Keyword limit: {keyword_limit}")
    print(f"   - Heading length: {max_heading_length} chars")
    print(f"   - Heading words: {max_heading_words}")
    print(f"   - Semantic weight: {semantic_weight}")
    print(f"   - TF-IDF weight: {tfidf_weight}")
    
    persona_analyzer = ConfigurablePersonaAnalyzer(keyword_limit, yake_top, yake_dedup)
    section_extractor = AdaptiveSectionExtractor(max_heading_length, max_heading_words, 
                                               min_paragraph_length, custom_prefixes)
    relevance_scorer = ConfigurableRelevanceScorer(model_name, tfidf_max_features, 
                                                  semantic_weight, tfidf_weight, batch_size)
    
    interest_profile = persona_analyzer.create_interest_profile(persona, job)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    all_sections = []
    for pdf_file in tqdm(pdf_files, desc="Extracting sections"):
        pdf_path = os.path.join(input_dir, pdf_file)
        sections = section_extractor.extract_sections(pdf_path)
        print(f"  📄 {pdf_file}: {len(sections)} sections extracted")
        all_sections.extend(sections)
    
    if not all_sections:
        print("⚠️  No sections extracted from PDFs")
        return
    
    print(f"📊 Total sections extracted: {len(all_sections)}")
    
    # Count sections per document
    doc_counts = {}
    for section in all_sections:
        doc = section['document']
        doc_counts[doc] = doc_counts.get(doc, 0) + 1
    
    print("📈 Sections per document:")
    for doc, count in doc_counts.items():
        print(f"  📄 {doc}: {count} sections")
    
    section_texts = [section['text'] for section in all_sections]
    scores = relevance_scorer.compute_scores(interest_profile, section_texts)
    
    for i, section in enumerate(all_sections):
        section['score'] = float(scores[i])
    
    all_sections.sort(key=lambda x: x['score'], reverse=True)
    
    # Ensure diversity in top sections
    top_sections = ensure_document_diversity(all_sections, top_k_sections)
    
    print(f"\n🏆 Top {len(top_sections)} sections (ensuring diversity):")
    for i, section in enumerate(top_sections):
        print(f"  {i+1}. {section['document']} - {section['title'][:50]}... (score: {section['score']:.3f})")
    
    # Subsection analysis
    subsection_analysis = []
    for section in top_sections:
        paras = [p.strip() for p in section['text'].split('\n\n') if len(p.strip()) > min_paragraph_length]
        if not paras:
            paras = [section['text']]
        
        sub_scores = relevance_scorer.compute_subsection_scores(interest_profile, paras)
        if len(sub_scores) > 0:
            best_idx = int(np.argmax(sub_scores))
            subsection_analysis.append({
                'document': section['document'],
                'refined_text': paras[best_idx],
                'page_number': section['page']
            })
    
    metadata = {
        'input_documents': pdf_files,
        'persona': persona,
        'job_to_be_done': job,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    extracted_sections = []
    for i, section in enumerate(top_sections):
        extracted_sections.append({
            'document': section['document'],
            'page_number': section['page'],
            'section_title': section['title'],
            'importance_rank': i + 1
        })
    
    result = {
        'metadata': metadata,
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 Extracted {len(extracted_sections)} top sections from {len(set(s['document'] for s in extracted_sections))} documents")
    print(f"🔍 Analyzed {len(subsection_analysis)} subsections")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round 1B: General PDF Processing Pipeline")
    parser.add_argument("--input_dir", default="./input", help="Input directory containing PDFs")
    parser.add_argument("--output_path", default="./output/general_result.json", help="Output JSON file path")
    parser.add_argument("--persona", required=True, help="Persona description")
    parser.add_argument("--job", required=True, help="Job to be done")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top sections to extract")
    
    # Keyword extraction parameters
    parser.add_argument("--keyword_limit", type=int, default=10, help="Maximum number of keywords to extract")
    parser.add_argument("--yake_top", type=int, default=20, help="YAKE top keywords to consider")
    parser.add_argument("--yake_dedup", type=float, default=0.3, help="YAKE deduplication limit")
    
    # Section extraction parameters
    parser.add_argument("--max_heading_length", type=int, default=200, help="Maximum heading length in characters")
    parser.add_argument("--max_heading_words", type=int, default=10, help="Maximum words in a heading")
    parser.add_argument("--min_paragraph_length", type=int, default=30, help="Minimum paragraph length")
    parser.add_argument("--custom_prefixes", nargs='+', help="Custom section prefixes to detect")
    
    # Scoring parameters
    # NEW, CORRECTED LINE
    parser.add_argument("--model_name", default="/model_files", help="Path to the local sentence transformer model")
    parser.add_argument("--tfidf_max_features", type=int, default=2000, help="TF-IDF maximum features")
    parser.add_argument("--semantic_weight", type=float, default=0.6, help="Semantic scoring weight")
    parser.add_argument("--tfidf_weight", type=float, default=0.4, help="TF-IDF scoring weight")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    
    args = parser.parse_args()
    main(args.input_dir, args.output_path, args.persona, args.job, args.top_k,
         args.keyword_limit, args.yake_top, args.yake_dedup,
         args.max_heading_length, args.max_heading_words, args.min_paragraph_length, args.custom_prefixes,
         args.model_name, args.tfidf_max_features, args.semantic_weight, args.tfidf_weight, args.batch_size) 