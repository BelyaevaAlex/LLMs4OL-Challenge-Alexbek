#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for computing TF-IDF with n-grams based on ontological terms from JSON files.
Updated version with part-of-speech analysis using NLTK.
"""

import json
import os
import re
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import math
from tqdm import tqdm

class TFIDFNgramsProcessor:
    """
    Class for processing documents and computing TF-IDF with n-grams
    based on ontological terms (OL) or document text.
    Includes part-of-speech analysis for extracting higher quality terms.
    """
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 3), use_pos_filtering: bool = True):
        """
        Initialize processor.
        
        Args:
            ngram_range: Range of n-grams (min_n, max_n)
            use_pos_filtering: Whether to use part-of-speech filtering
        """
        self.ngram_range = ngram_range
        self.min_n, self.max_n = ngram_range
        self.document_frequency = defaultdict(int)  # Document frequency for each n-gram
        self.total_documents = 0
        
        # Initialize NLTK
        self.nltk_available = False
        self.use_pos_filtering = use_pos_filtering
        
        if use_pos_filtering:
            try:
                import nltk
                from nltk.tokenize import word_tokenize, sent_tokenize
                from nltk.tag import pos_tag
                from nltk.corpus import stopwords
                
                self.nltk = nltk
                self.word_tokenize = word_tokenize
                self.sent_tokenize = sent_tokenize
                self.pos_tag = pos_tag
                
                try:
                    self.stop_words = set(stopwords.words('english'))
                except:
                    self.stop_words = set()
                
                self.nltk_available = True
                print("✓ NLTK loaded successfully for part-of-speech analysis")
            except ImportError as e:
                print(f"⚠ NLTK not available ({e}), working without part-of-speech analysis")
                self.nltk_available = False
        else:
            print("Part-of-speech analysis disabled")
        
        # Excluded parts of speech (based on OL analysis)
        self.excluded_pos = {
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'RB', 'RBR', 'RBS',  # Adverbs
            'PRP', 'PRP$',  # Pronouns
            'JJR', 'JJS'  # Comparative and superlative adjectives
        }
        
        # Important parts of speech for preservation (based on OL analysis)
        self.important_pos = {
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'JJ',  # Adjectives (positive degree only)
            'CD',  # Cardinal numbers
            'FW'   # Foreign words
        }
        
        # Updated list of prohibited words (based on analysis)
        self.prohibited_words = {
            'into', 'familiar', 'addition', 'diverse', 'common', 'extent', 
            'notable', 'complex', 'various', 'different', 'several', 'many',
            'some', 'other', 'such', 'most', 'more', 'less', 'much', 'very',
            'quite', 'rather', 'really', 'actually', 'generally', 'usually',
            'often', 'sometimes', 'always', 'never', 'also', 'even', 'still',
            'just', 'only', 'mainly', 'mostly', 'especially', 'particularly',
            # New stop words based on OL analysis
            'of', 'in', 'from', 'with', 'by', 'for', 'to', 'and', 'or', 'the',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Technical prefixes to preserve (based on OL analysis)
        self.technical_prefixes = {
            'bio', 'eco', 'hydro', 'micro', 'nano', 'macro', 'geo', 
            'thermo', 'photo', 'electro', 'neuro', 'cardio', 'gastro',
            'hepato', 'nephro', 'pneumo', 'osteo', 'myco', 'crypto'
        }
        
        # Semantic categories for prioritization (based on OL analysis)
        self.priority_categories = {
            'biological': ['bio', 'cell', 'organism', 'plant', 'animal', 'gene', 
                          'protein', 'enzyme', 'tissue', 'organ', 'species', 
                          'population', 'ecosystem', 'biome', 'flora', 'fauna'],
            'chemical': ['compound', 'molecule', 'atom', 'ion', 'acid', 'base', 
                        'salt', 'oxide', 'carbon', 'nitrogen', 'oxygen', 'hydrogen', 
                        'chemical', 'reaction', 'solution'],
            'geographical': ['lake', 'river', 'mountain', 'forest', 'desert', 
                           'ocean', 'sea', 'island', 'continent', 'region', 'area', 
                           'zone', 'location', 'place']
        }
        
        # Parts of speech that we EXCLUDE (verbs, adverbs, participles, pronouns, comparative degrees)
        self.excluded_words = {
            # Prepositions and conjunctions
            'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'as', 'an', 'a', 'the',
            'and', 'or', 'but', 'if', 'then', 'than', 'when', 'where', 'while', 'during', 'through',
            'into', 'addition',  # Added new words
            
            # State verbs and linking verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            
            # Adverbs and problematic general words
            'very', 'quite', 'rather', 'too', 'so', 'just', 'only', 'even', 'also', 'still', 'yet',
            'already', 'again', 'once', 'twice', 'always', 'never', 'often', 'sometimes', 'usually',
            'certain', 'harmful', 'vital', 'various', 'right', 'different', 'vary', 'crucial', 'significant',
            'fundamental', 'other', 'dry', 'rich', 'wet', 'visible', 'specific', 'full', 'adverse', 'instance',
            'belong', 'due', 'per', 'equal', 'initial', 'scientific', 'utilize', 'utilizes', 'utilized',
            'utilizing', 'vast', 'describe', 'describes', 'described', 'describing', 'familiar',  # Added new word
            'diverse', 'common', 'extent', 'notable', 'complex',  # Added new prohibited words
            
            # Comparative and superlative degrees of adjectives
            'better', 'best', 'worse', 'worst', 'bigger', 'biggest', 'smaller', 'smallest', 'larger', 'largest',
            'higher', 'highest', 'lower', 'lowest', 'longer', 'longest', 'shorter', 'shortest', 'older', 'oldest',
            'newer', 'newest', 'younger', 'youngest', 'faster', 'fastest', 'slower', 'slowest', 'stronger', 'strongest',
            'weaker', 'weakest', 'easier', 'easiest', 'harder', 'hardest', 'simpler', 'simplest', 'more', 'most',
            'less', 'least', 'greater', 'greatest', 'lesser', 'wider', 'widest', 'narrower', 'narrowest',
            'deeper', 'deepest', 'thicker', 'thickest', 'thinner', 'thinnest', 'closer', 'closest', 'further', 'furthest',
            'nearer', 'nearest', 'earlier', 'earliest', 'later', 'latest', 'richer', 'richest', 'poorer', 'poorest',
            
            # Colors
            'blue', 'red', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey', 'pink', 'purple',
            'orange', 'violet', 'indigo', 'cyan', 'magenta', 'silver', 'gold', 'dark', 'light',
            
            # Adjectives
            'good', 'bad', 'big', 'small', 'large', 'little', 'long', 'short', 'high', 'low', 'old', 'new',
            'young', 'early', 'late', 'first', 'last', 'next', 'previous', 'same', 'different', 'similar',
            'important', 'main', 'major', 'minor', 'great', 'small', 'large', 'huge', 'tiny', 'massive',
            
            # Service words
            'this', 'that', 'these', 'those', 'here', 'there', 'now', 'then', 'today', 'yesterday',
            'tomorrow', 'yes', 'no', 'not', 'none', 'all', 'some', 'any', 'each', 'every', 'both',
            'either', 'neither', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
        }
        
        # For backward compatibility, keep old variables as references to the new list
        self.bad_start_words = self.excluded_words
        self.bad_end_words = self.excluded_words
        self.bad_endings = {'ly', 'er', 'est', 'ing', 'ed', 'es', 's'}
        
        # Important parts of speech (patterns for simple determination)
        self.important_patterns = {
            'noun_endings': ['tion', 'sion', 'ness', 'ment', 'ity', 'ism', 'er', 'or', 'ant', 'ent'],
            'adj_endings': ['al', 'ic', 'ous', 'ful', 'less', 'able', 'ible', 'ive', 'ary', 'ory'],
            'tech_prefixes': ['bio', 'geo', 'eco', 'micro', 'macro', 'nano', 'mega', 'giga', 'tera', 'hydro', 'thermo'],
            'tech_suffixes': ['ology', 'ography', 'metry', 'scopy', 'logy', 'graphy', 'phobia', 'philia']
        }
        
        # Verb endings (to exclude verbs)
        self.verb_endings = ['ing', 'ed', 'es', 's', 'en', 'ate', 'ize', 'ise', 'fy']
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text with improved tokenization.
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace punctuation with spaces (so words don't stick together)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces at beginning and end
        text = text.strip()
        
        return text
    
    def simple_tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization of text into sentences and words.
        
        Args:
            text: Original text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Split into sentences by periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if cleaned and len(cleaned.split()) >= self.min_n:
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    def is_meaningful_word(self, word: str) -> bool:
        """
        Checks if a word is meaningful for term extraction.
        
        Args:
            word (str): Word to check
            
        Returns:
            bool: True if word is meaningful, False otherwise
        """
        if not word or len(word) < 2:
            return False
        
        word_lower = word.lower()
        
        # Exclude words from combined exclusion list
        if word_lower in self.excluded_words:
            return False
        
        # Exclude comparative and superlative degrees of adjectives by endings
        comparative_endings = ['er', 'est']
        if any(word_lower.endswith(ending) for ending in comparative_endings):
            # Check if this is not a technical term
            if not self.is_technical_term(word):
                # Additional check: if word is short and ends with er/est, it's likely a comparative degree
                if len(word) <= 6:
                    return False
        
        # Stricter verb checking by endings
        if any(word_lower.endswith(ending) for ending in self.verb_endings):
            # Exception for technical terms
            if not self.is_technical_term(word):
                return False
        
        # Exclude monosyllabic functional words
        if len(word) <= 3 and word_lower in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}:
            return False
        
        # Exclude words consisting only of digits
        if word.isdigit():
            return False
        
        # Exclude words with excessive special characters
        special_char_count = sum(1 for c in word if not c.isalnum())
        if special_char_count > len(word) * 0.3:
            return False
        
        # Technical terms are always considered meaningful
        if self.is_technical_term(word):
            return True
        
        return True
    
    def is_technical_term(self, word: str) -> bool:
        """
        Checks if a word is a technical term.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is a technical term
        """
        word_lower = word.lower()
        
        # Check technical prefixes
        for prefix in self.important_patterns['tech_prefixes']:
            if word_lower.startswith(prefix):
                return True
        
        # Check technical suffixes
        for suffix in self.important_patterns['tech_suffixes']:
            if word_lower.endswith(suffix):
                return True
        
        # Check noun and adjective endings
        for ending in self.important_patterns['noun_endings'] + self.important_patterns['adj_endings']:
            if word_lower.endswith(ending):
                return True
        
        return False
    
    def extract_meaningful_phrases(self, text: str) -> List[str]:
        """
        Extract meaningful phrases using NLTK part-of-speech analysis.
        Excludes verbs, adverbs, participles, and pronouns.
        
        Args:
            text: Original text
            
        Returns:
            List of meaningful phrases
        """
        if not self.use_pos_filtering or not self.nltk_available:
            # If NLTK is not available, use simple approach
            return self._extract_phrases_simple(text)
        
        try:
            # Tokenize into sentences
            sentences = self.sent_tokenize(text)
            phrases = []
            
            for sentence in sentences:
                # Tokenize into words
                words = self.word_tokenize(sentence.lower())
                
                # Remove punctuation and short words
                words = [word for word in words if word.isalpha() and len(word) > 2]
                
                if not words:
                    continue
                
                # POS tagging
                pos_tags = self.pos_tag(words)
                
                # Extract sequences of important parts of speech
                current_phrase = []
                for word, pos in pos_tags:
                    # EXCLUDE words with unwanted parts of speech
                    if pos in self.excluded_pos:
                        # Complete current phrase if it exists
                        if len(current_phrase) >= self.min_n:
                            phrase = ' '.join(current_phrase)
                            if len(current_phrase) <= self.max_n and self.is_valid_phrase(phrase):
                                phrases.append(phrase)
                        current_phrase = []
                        continue
                    
                    # EXCLUDE words from excluded_words list
                    if word.lower() in self.excluded_words:
                        # Complete current phrase if it exists
                        if len(current_phrase) >= self.min_n:
                            phrase = ' '.join(current_phrase)
                            if len(current_phrase) <= self.max_n and self.is_valid_phrase(phrase):
                                phrases.append(phrase)
                        current_phrase = []
                        continue
                    
                    # SAVE only words with important parts of speech
                    if pos in self.important_pos and word not in self.stop_words:
                        # EXCLUDE 1-grams of adjectives (only JJ)
                        if pos == 'JJ' and len(current_phrase) == 0 and self.min_n == 1:
                            # Skip single adjectives if min_n = 1
                            continue
                        current_phrase.append(word)
                    else:
                        # Complete current phrase if it exists
                        if len(current_phrase) >= self.min_n:
                            phrase = ' '.join(current_phrase)
                            if len(current_phrase) <= self.max_n and self.is_valid_phrase(phrase):
                                phrases.append(phrase)
                        current_phrase = []
                
                # Add last phrase if it exists
                if len(current_phrase) >= self.min_n and len(current_phrase) <= self.max_n:
                    phrase = ' '.join(current_phrase)
                    # ADDITIONAL CHECK: exclude 1-gram adjectives
                    if len(current_phrase) == 1 and pos_tags:
                        # Find POS tag for last word
                        last_word_pos = None
                        for w, p in pos_tags:
                            if w == current_phrase[0]:
                                last_word_pos = p
                                break
                        # Exclude single adjectives
                        if last_word_pos == 'JJ':
                            continue
                    
                    if self.is_valid_phrase(phrase):
                        phrases.append(phrase)
            
            # If nothing found with NLTK, use simple approach
            if not phrases:
                return self._extract_phrases_simple(text)
            
            return list(set(phrases))  # Remove duplicates
            
        except Exception as e:
            print(f"Part-of-speech analysis error for '{text[:50]}...': {e}")
            return self._extract_phrases_simple(text)
    
    def _extract_phrases_simple(self, text: str) -> List[str]:
        """
        Simple phrase extraction without NLTK (fallback method).
        
        Args:
            text: Original text
            
        Returns:
            List of phrases
        """
        if not text:
            return []
        
        # Tokenize into sentences
        sentences = self.simple_tokenize(text)
        phrases = []
        
        for sentence in sentences:
            words = sentence.split()
            
            # Filter meaningful words
            meaningful_words = []
            for word in words:
                if self.is_meaningful_word(word) or self.is_technical_term(word):
                    meaningful_words.append(word)
            
            # Create phrases from consecutive meaningful words
            if len(meaningful_words) >= self.min_n:
                # Generate sliding windows
                for i in range(len(meaningful_words)):
                    for j in range(self.min_n, min(self.max_n + 1, len(meaningful_words) - i + 1)):
                        phrase = ' '.join(meaningful_words[i:i+j])
                        # ADD PHRASE VALIDITY CHECK
                        if phrase and self.is_valid_phrase(phrase):
                            phrases.append(phrase)
        
        # Remove duplicates and return
        return list(set(phrases)) if phrases else []
    
    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Generate n-grams from list of tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-gram
            
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def extract_ngrams_from_ol(self, ol_terms: List[str]) -> List[str]:
        """
        Extract n-grams from ontological terms (OL) with part-of-speech analysis.
        Returns only maximally complete and meaningful n-grams.
        
        Args:
            ol_terms: List of ontological terms
            
        Returns:
            List of unique and meaningful n-grams
        """
        all_ngrams = set()
        
        for term in ol_terms:
            if not term or not term.strip():
                continue
            
            # First try to extract meaningful phrases
            meaningful_phrases = self.extract_meaningful_phrases(term)
            
            for phrase in meaningful_phrases:
                if phrase:
                    all_ngrams.add(phrase)
            
            # Additionally generate n-grams from cleaned term
            cleaned_term = self.clean_text(term)
            if cleaned_term:
                tokens = cleaned_term.split()
                
                # Generate n-grams for each size
                for n in range(self.min_n, min(len(tokens) + 1, self.max_n + 1)):
                    ngrams = self.generate_ngrams(tokens, n)
                    all_ngrams.update(ngrams)
        
        # Filter: keep only maximally complete n-grams
        filtered_ngrams = self.filter_maximal_ngrams(list(all_ngrams))
        
        return filtered_ngrams
    
    def extract_ngrams_from_text(self, text: str, ngram_range: Tuple[int, int] = None) -> List[str]:
        """
        Extract n-grams from text with part-of-speech analysis.
        Excludes verbs, adverbs, participles, and pronouns using NLTK.
        
        Args:
            text: Original text
            ngram_range: Range of n-grams (min, max). If None, uses self.ngram_range
            
        Returns:
            List of n-grams
        """
        if not text.strip():
            return []
        
        # Determine n-gram range
        if ngram_range is None:
            min_n, max_n = self.min_n, self.max_n
        else:
            min_n, max_n = ngram_range
        
        # First try to extract meaningful phrases using NLTK
        if self.use_pos_filtering and self.nltk_available:
            try:
                meaningful_phrases = self.extract_meaningful_phrases(text)
                if meaningful_phrases:
                    # Generate n-grams from meaningful phrases
                    all_ngrams = []
                    for phrase in meaningful_phrases:
                        tokens = phrase.split()
                        # Generate n-grams for all sizes from min_n to max_n
                        for n in range(min_n, min(len(tokens) + 1, max_n + 1)):
                            ngrams = self.generate_ngrams(tokens, n)
                            # Additionally filter each n-gram
                            valid_ngrams = [ngram for ngram in ngrams if self.is_valid_phrase(ngram)]
                            all_ngrams.extend(valid_ngrams)
                    
                    if all_ngrams:
                        return list(set(all_ngrams))  # Remove duplicates
            except Exception as e:
                print(f"Part-of-speech analysis error: {e}")
        
        # If part-of-speech analysis doesn't work, use simple approach
        return self._extract_ngrams_simple(text, ngram_range)
    
    def _extract_ngrams_simple(self, text: str, ngram_range: Tuple[int, int] = None) -> List[str]:
        """
        Simple n-gram extraction without NLTK (fallback method).
        
        Args:
            text: Original text
            ngram_range: Range of n-grams (min, max). If None, uses self.ngram_range
            
        Returns:
            List of n-grams
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        # Determine n-gram range
        if ngram_range is None:
            min_n, max_n = self.min_n, self.max_n
        else:
            min_n, max_n = ngram_range
        
        # Split into sentences for better n-gram extraction
        sentences = cleaned_text.split('.')
        all_ngrams = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                tokens = sentence.split()
                # Generate n-grams for all sizes from min_n to max_n
                for n in range(min_n, min(len(tokens) + 1, max_n + 1)):
                    ngrams = self.generate_ngrams(tokens, n)
                    # Filter correct phrases
                    valid_ngrams = [ngram for ngram in ngrams if self.is_valid_phrase(ngram)]
                    all_ngrams.extend(valid_ngrams)
        
        return list(set(all_ngrams))  # Remove duplicates
    
    def filter_maximal_ngrams(self, ngrams: List[str]) -> List[str]:
        """
        Filters n-grams, keeping only maximally complete ones
        (removes those that are substrings of others).
        
        Args:
            ngrams: List of all n-grams
            
        Returns:
            List of maximally complete n-grams
        """
        if not ngrams:
            return []
        
        # Remove duplicates and sort by length (from long to short)
        unique_ngrams = list(set(ngrams))
        sorted_ngrams = sorted(unique_ngrams, key=lambda x: (len(x.split()), len(x)), reverse=True)
        maximal_ngrams = []
        
        for ngram in sorted_ngrams:
            # Check if current n-gram is a substring of already added ones
            is_substring = False
            for existing_ngram in maximal_ngrams:
                # Check if ngram is contained as a substring in existing_ngram
                if ngram != existing_ngram and ngram in existing_ngram:
                    # Additional check: ngram must be separate words
                    ngram_words = set(ngram.split())
                    existing_words = set(existing_ngram.split())
                    if ngram_words.issubset(existing_words):
                        is_substring = True
                        break
            
            if not is_substring:
                maximal_ngrams.append(ngram)
        
        return maximal_ngrams
    
    def compute_tf(self, ngrams: List[str]) -> Dict[str, float]:
        """
        Compute Term Frequency (TF) for document n-grams.
        
        Args:
            ngrams: List of document n-grams
            
        Returns:
            Dictionary with TF values for each n-gram
        """
        if not ngrams:
            return {}
            
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        tf_scores = {}
        for ngram, count in ngram_counts.items():
            tf_scores[ngram] = count / total_ngrams
            
        return tf_scores
    
    def compute_idf(self) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency (IDF) for all n-grams.
        
        Returns:
            Dictionary with IDF values for each n-gram
        """
        idf_scores = {}
        for ngram, doc_freq in self.document_frequency.items():
            idf_scores[ngram] = math.log(self.total_documents / doc_freq)
            
        return idf_scores
    
    def compute_tfidf(self, tf_scores: Dict[str, float], idf_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute TF-IDF values.
        
        Args:
            tf_scores: TF values for document
            idf_scores: IDF values for all n-grams
            
        Returns:
            Dictionary with TF-IDF values
        """
        tfidf_scores = {}
        for ngram, tf in tf_scores.items():
            if ngram in idf_scores:
                tfidf_scores[ngram] = tf * idf_scores[ngram]
                
        return tfidf_scores
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process all documents to compute TF-IDF.
        Now always uses document text to extract n-grams.
        
        Args:
            documents: List of documents
            
        Returns:
            List of documents with added TF-IDF values in term list format
        """
        print(f"Part-of-speech analysis disabled" if not self.use_pos_filtering else f"✓ NLTK loaded successfully for part-of-speech analysis")
        
        # First pass: extract all n-grams and count document frequency
        all_documents_ngrams = []
        
        for doc in tqdm(documents, desc="Extracting n-grams"):
            # Always use document text to extract n-grams
            text_content = ""
            
            # Collect all available text
            if 'title' in doc and doc['title']:
                text_content += doc['title'] + ". "
            
            if 'text' in doc and doc['text']:
                text_content += doc['text']
            
            # If OL exists, add it as additional context
            if 'OL' in doc and doc['OL']:
                ol_text = ". ".join(doc['OL'])
                text_content += ". " + ol_text
            
            if not text_content.strip():
                print(f"⚠ Empty document: {doc.get('id', 'unknown ID')}")
                all_documents_ngrams.append([])
                continue
            
            # Extract n-grams from all text
            ngrams = self.extract_ngrams_from_text(text_content)
            all_documents_ngrams.append(ngrams)
            
            # Update document frequency
            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                self.document_frequency[ngram] += 1
        
        self.total_documents = len(documents)
        print(f"Processed documents: {self.total_documents}")
        print(f"Unique n-grams found: {len(self.document_frequency)}")
        
        # Compute IDF
        idf_scores = self.compute_idf()
        
        # Second pass: compute TF-IDF for each document
        processed_documents = []
        documents_with_tfidf = 0
        
        for i, doc in enumerate(tqdm(documents, desc="Computing TF-IDF")):
            ngrams = all_documents_ngrams[i]
            
            if not ngrams:
                # Document without n-grams
                processed_doc = doc.copy()
                processed_doc['TF-IDF'] = []
                processed_documents.append(processed_doc)
                continue
            
            # Compute TF
            tf_scores = self.compute_tf(ngrams)
            
            # Compute TF-IDF
            tfidf_scores = self.compute_tfidf(tf_scores, idf_scores)
            
            # Get top terms with improved filtering
            top_terms = self.get_top_tfidf_terms(tfidf_scores, max_terms=12)
            
            # Add TF-IDF data to document
            doc_copy = doc.copy()
            doc_copy['TF-IDF'] = top_terms
            processed_documents.append(doc_copy)
            
            if top_terms:
                documents_with_tfidf += 1
        
        print(f"Documents with TF-IDF: {documents_with_tfidf}")
        return processed_documents

    def is_valid_phrase(self, phrase: str) -> bool:
        """
        Validate phrase considering OL field analysis.
        """
        if not phrase or len(phrase.strip()) < 2:
            return False
        
        phrase = phrase.strip().lower()
        
        # Exclude prohibited words
        if phrase in self.prohibited_words:
            return False
        
        # Exclude phrases consisting only of prohibited words
        words = phrase.split()
        if all(word in self.prohibited_words for word in words):
            return False
        
        # Exclude too short phrases
        if len(words) == 1 and len(phrase) < 3:
            return False
        
        # Exclude phrases consisting only of digits and punctuation
        if re.match(r'^[\d\s\W]+$', phrase):
            return False
        
        # Check using NLTK (if available)
        if self.nltk_available:
            try:
                # Exclude 1-gram adjectives (except technical ones)
                if len(words) == 1:
                    pos_tags = self.pos_tag([phrase])
                    if pos_tags and pos_tags[0][1] == 'JJ':
                        # Check if this is a technical term
                        if not any(phrase.startswith(prefix) for prefix in self.technical_prefixes):
                            return False
                
                # Analyze entire phrase
                pos_tags = self.pos_tag(words)
                
                # Exclude phrases with unwanted POS patterns
                pos_sequence = [tag for _, tag in pos_tags]
                
                # Exclude phrases starting with verbs
                if pos_sequence and pos_sequence[0] in self.excluded_pos:
                    return False
                
                # Exclude phrases with predominance of excluded POS
                excluded_count = sum(1 for pos in pos_sequence if pos in self.excluded_pos)
                if excluded_count > len(pos_sequence) / 2:
                    return False
                
                # Prioritize phrases with important POS
                important_count = sum(1 for pos in pos_sequence if pos in self.important_pos)
                if important_count == 0 and len(pos_sequence) > 1:
                    return False
                
            except Exception:
                pass  # If NLTK doesn't work, continue without POS analysis
        
        # Check semantic category (bonus for priority ones)
        is_priority = False
        for category, keywords in self.priority_categories.items():
            if any(keyword in phrase for keyword in keywords):
                is_priority = True
                break
        
        # Stricter rules for non-priority terms
        if not is_priority:
            # Exclude very short non-priority terms
            if len(phrase) < 4:
                return False
            
            # Exclude too generic words
            generic_words = {'thing', 'item', 'object', 'element', 'part', 'piece', 'type', 'kind', 'sort'}
            if any(word in generic_words for word in words):
                return False
        
        return True

    def get_top_tfidf_terms(self, tfidf_scores, max_terms=12):
        """
        Returns top terms by TF-IDF with improved filtering.
        
        Args:
            tfidf_scores: Dictionary with TF-IDF values
            max_terms: Maximum number of terms (default 12)
            
        Returns:
            List of best terms
        """
        if not tfidf_scores:
            return []
        
        # Sort by descending TF-IDF
        sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Additional filtering of quality terms
        quality_terms = []
        for term, score in sorted_terms:
            # Skip too short terms
            if len(term) < 3:
                continue
            
            # Skip terms with very low TF-IDF
            if score < 0.01:
                continue
            
            words = term.split()
            if not words:
                continue
            
            # IMPORTANT: Check that NO word in the term is bad
            has_bad_word = False
            for word in words:
                word_lower = word.lower()
                if word_lower in self.excluded_words:
                    has_bad_word = True
                    break
            
            if has_bad_word:
                continue
            
            # Check that term doesn't end with bad words
            last_word = words[-1].lower()
            if last_word in self.excluded_words:
                continue
            
            # NEW: Check that term doesn't end with bad endings
            if any(last_word.endswith(ending) for ending in self.bad_endings):
                # Exception: if last word is a technical term
                if not self.is_technical_term(last_word):
                    continue
            
            # Check that term doesn't consist only of stop words
            meaningful_words = [w for w in words if w.lower() not in self.excluded_words and 
                               w.lower() not in self.bad_endings]
            if len(meaningful_words) == 0:
                continue
            
            # Check proportion of meaningful words in term
            if len(meaningful_words) / len(words) < 0.5:
                continue
            
            # Use general phrase validity check
            if not self.is_valid_phrase(term):
                continue
            
            quality_terms.append((term, score))
            
            # Limit number of terms
            if len(quality_terms) >= max_terms:
                break
        
        # Return only terms (without scores)
        return [term for term, score in quality_terms]

    def find_jsonl_files(self, input_dir: str) -> List[str]:
        """
        Finds all JSONL files only in test and train folders (not recursively).
        Looks for files named docs2terms.jsonl or text2onto_*_test_documents.jsonl
        
        Args:
            input_dir: Input directory
            
        Returns:
            List of paths to JSONL files
        """
        jsonl_files = []
        
        # Walk through all subfolders in input_dir
        for root, dirs, files in os.walk(input_dir):
            # Check that we are in test or train folder
            if os.path.basename(root) in ['test', 'train']:
                for file in files:
                    if file.endswith('.jsonl'):
                        # Check filename
                        if (file == 'docs2terms.jsonl' or 
                            file.startswith('text2onto_') and file.endswith('_test_documents.jsonl')):
                            file_path = os.path.join(root, file)
                            jsonl_files.append(file_path)
        
        return sorted(jsonl_files)

    def calculate_tfidf_scores(self, phrases_by_doc: List[List[str]]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores considering semantic categories and complexity.
        """
        if not phrases_by_doc:
            return {}
        
        # Count term frequencies
        term_freq = Counter()
        doc_freq = Counter()
        
        for doc_phrases in phrases_by_doc:
            doc_terms = set()
            for phrase in doc_phrases:
                term_freq[phrase] += 1
                doc_terms.add(phrase)
            
            for term in doc_terms:
                doc_freq[term] += 1
        
        # Calculate TF-IDF with modifications
        tfidf_scores = {}
        total_docs = len(phrases_by_doc)
        
        for term in term_freq:
            tf = term_freq[term]
            df = doc_freq[term]
            
            # Base TF-IDF
            idf = math.log(total_docs / df) if df > 0 else 0
            base_score = tf * idf
            
            # Modifiers based on OL analysis
            modifier = 1.0
            
            # Bonus for technical prefixes
            if any(term.startswith(prefix) for prefix in self.technical_prefixes):
                modifier *= 1.3
            
            # Bonus for priority semantic categories
            for category, keywords in self.priority_categories.items():
                if any(keyword in term.lower() for keyword in keywords):
                    if category == 'chemical':
                        modifier *= 1.25  # Chemical terms - highest priority
                    elif category == 'biological':
                        modifier *= 1.2   # Biological terms
                    elif category == 'geographical':
                        modifier *= 1.15  # Geographical terms
                    break
            
            # Complexity modifier
            words = term.split()
            word_count = len(words)
            
            if word_count == 1:
                # Bonus for simple but important terms
                if len(term) >= 6 and any(term.startswith(prefix) for prefix in self.technical_prefixes):
                    modifier *= 1.1
                else:
                    modifier *= 0.9  # Small penalty for simple terms
            elif word_count == 2:
                # Optimal length for most terms
                modifier *= 1.05
            elif word_count >= 5:
                # Penalty for complexity
                modifier *= 0.85
            
            # Penalty for too high frequency (possibly common words)
            if tf > total_docs * 0.1:  # If term appears in >10% of documents
                modifier *= 0.8
            
            # Bonus for moderate frequency
            if 2 <= tf <= total_docs * 0.05:
                modifier *= 1.1
            
            tfidf_scores[term] = base_score * modifier
        
        return tfidf_scores

    def process_document(self, doc: Dict, ngram_range: Tuple[int, int] = (1, 3)) -> List[str]:
        """
        Process single document with improved term extraction.
        """
        if 'text' not in doc or not doc['text']:
            return []
        
        text = doc['text']
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Extract meaningful phrases using NLTK
        meaningful_phrases = self.extract_meaningful_phrases(text)
        
        # Extract n-grams from text
        ngrams = self.extract_ngrams_from_text(text, ngram_range)
        
        # Combine and filter
        all_candidates = list(set(meaningful_phrases + ngrams))
        
        # Filter by validity
        valid_phrases = [phrase for phrase in all_candidates if self.is_valid_phrase(phrase)]
        
        return valid_phrases

    def process_documents_batch(self, documents: List[Dict], ngram_range: Tuple[int, int] = (1, 6)) -> Tuple[List[List[str]], Dict[str, float]]:
        """
        Process batch of documents with TF-IDF computation.
        """
        print(f"Processing {len(documents)} documents...")
        
        all_phrases_by_doc = []
        
        for i, doc in enumerate(documents):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents")
            
            phrases = self.process_document(doc, ngram_range)
            all_phrases_by_doc.append(phrases)
        
        # Calculate TF-IDF scores
        print("Calculating TF-IDF scores...")
        tfidf_scores = self.calculate_tfidf_scores(all_phrases_by_doc)
        
        return all_phrases_by_doc, tfidf_scores

    def save_results_with_tfidf(self, input_file: str, output_file: str, phrases_by_doc: List[List[str]], 
                               tfidf_scores: Dict[str, float], top_k: int = 50):
        """
        Save results with TF-IDF terms (without scores).
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            phrases_by_doc: List of phrases for each document
            tfidf_scores: TF-IDF scores for terms
            top_k: Number of top terms for each document
        """
        try:
            # Load original documents
            documents = load_jsonl_file(input_file)
            
            if len(documents) != len(phrases_by_doc):
                print(f"Warning: number of documents ({len(documents)}) doesn't match number of processed ({len(phrases_by_doc)})")
                min_len = min(len(documents), len(phrases_by_doc))
                documents = documents[:min_len]
                phrases_by_doc = phrases_by_doc[:min_len]
            
            # Enrich documents with TF-IDF terms
            for i, (doc, phrases) in enumerate(zip(documents, phrases_by_doc)):
                # Get scores for phrases of this document
                doc_tfidf_scores = {}
                for phrase in phrases:
                    if phrase in tfidf_scores:
                        doc_tfidf_scores[phrase] = tfidf_scores[phrase]
                
                # Sort by score and take top-k
                sorted_terms = sorted(doc_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
                top_terms = [term for term, score in sorted_terms[:top_k]]
                
                # If terms are few, add remaining phrases
                if len(top_terms) < top_k:
                    remaining_phrases = [p for p in phrases if p not in top_terms]
                    top_terms.extend(remaining_phrases[:top_k - len(top_terms)])
                
                # Add TF-IDF field (only list of terms without scores)
                doc['TF-IDF'] = top_terms
            
            # Save results
            save_jsonl_file(documents, output_file)
            
        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

def load_jsonl_file(file_path: str) -> List[Dict]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of documents
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        doc = json.loads(line)
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error in line {line_num} of file {file_path}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    return documents

def save_jsonl_file(documents: List[Dict], file_path: str) -> None:
    """
    Save documents to JSONL file.
    
    Args:
        documents: List of documents
        file_path: Path for saving
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process JSONL files with improved TF-IDF analysis')
    # parser.add_argument('--input_dir', type=str, help='Directory with input JSONL files')
    # parser.add_argument('--output_dir', type=str, help='Directory for output files')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=6, help='Maximum n-gram size')
    parser.add_argument('--top_k', type=int, default=50, help='Number of top terms per document')
    
    args = parser.parse_args()
    
    processor = TFIDFNgramsProcessor()
    
    subdirs = ['scholarly', 'engineering', 'ecology']
    for subdir in subdirs:
        input_dir = f'../../2025/TaskA-Text2Onto/{subdir}/'
        output_dir = f'../../2025/TaskA-Text2Onto-Processed/{subdir}/'

    
        # Find JSONL files
        jsonl_files = processor.find_jsonl_files(input_dir)
        
        if not jsonl_files:
            print("JSONL files not found!")
            return
        
        print(f"Found {len(jsonl_files)} JSONL files to process")
        for file_path in jsonl_files:
            print(f"  - {file_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        total_documents = 0
        
        # Process each file
        for file_path in jsonl_files:
            print(f"\n{'='*80}")
            print(f"Processing file: {file_path}")
            
            # Load documents
            documents = load_jsonl_file(file_path)
            if not documents:
                print(f"File {file_path} is empty or cannot be loaded")
                continue
            
            print(f"Loaded documents: {len(documents)}")
            total_documents += len(documents)
            
            # Process documents with TF-IDF
            phrases_by_doc, tfidf_scores = processor.process_documents_batch(
                documents, 
                ngram_range=(args.ngram_min, args.ngram_max)
            )
            
            # Determine output file
            relative_path = os.path.relpath(file_path, input_dir)
            output_file = os.path.join(output_dir, relative_path)
            
            # Create directory for output file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save results
            processor.save_results_with_tfidf(
                file_path, 
                output_file, 
                phrases_by_doc, 
                tfidf_scores, 
                args.top_k
            )
            
            print(f"Results saved to: {output_file}")
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {subdir} COMPLETED")
        print(f"Total documents processed: {total_documents}")
        print(f"Results saved to: {output_dir}")
        print(f"Field 'TF-IDF' added to each document (list of terms without scores)")

if __name__ == "__main__":
    main() 