"""Query preprocessing service for enhanced embedding generation."""

import re
import logging
from typing import List, Set, Tuple
import nltk
import spacy
from app.config import settings

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Service for preprocessing queries to extract keywords and remove noise."""
    
    def __init__(self):
        """Initialize preprocessor with NLTK and spaCy models."""
        self.nlp = None
        self.stop_words = set()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLTK and spaCy models."""
        try:
            # Download required NLTK data
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Load stopwords
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
            
            # Add custom stopwords for travel/literature context
            custom_stopwords = {
                'want', 'would', 'like', 'know', 'tell', 'me', 'about', 'please',
                'can', 'could', 'should', 'will', 'shall', 'may', 'might',
                'what', 'where', 'when', 'why', 'how', 'which', 'who',
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
                'through', 'during', 'before', 'after', 'above', 'below',
                'between', 'among', 'around', 'near', 'far', 'here', 'there',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                'it', 'we', 'they', 'them', 'their', 'his', 'her', 'its',
                'my', 'your', 'our', 'mine', 'yours', 'ours', 'theirs'
            }
            self.stop_words.update(custom_stopwords)
            
            logger.info("Query preprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing query preprocessor: {e}")
            raise
    
    def preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Preprocess query to extract keywords and create cleaned version.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple of (cleaned_query, extracted_keywords)
        """
        try:
            # Clean the query
            cleaned_query = self._clean_text(query)
            
            # Extract keywords using multiple methods
            keywords = self._extract_keywords(cleaned_query)
            
            # Create keyword-focused query
            keyword_query = " ".join(keywords) if keywords else cleaned_query
            
            logger.debug(f"Original query: {query}")
            logger.debug(f"Cleaned query: {cleaned_query}")
            logger.debug(f"Keywords: {keywords}")
            logger.debug(f"Keyword query: {keyword_query}")
            
            return keyword_query, keywords
            
        except Exception as e:
            logger.error(f"Error preprocessing query: {e}")
            # Fallback to basic cleaning
            return self._basic_clean(query), []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using NER and linguistic analysis."""
        keywords = set()
        
        # Method 1: Named Entity Recognition (if spaCy available)
        if self.nlp:
            keywords.update(self._extract_entities(text))
        
        # Method 2: Important words (nouns, verbs, adjectives)
        keywords.update(self._extract_important_words(text))
        
        # Method 3: Travel/location specific terms
        keywords.update(self._extract_travel_terms(text))
        
        # Filter out stopwords and short words
        filtered_keywords = [
            word for word in keywords
            if len(word) > 2 and word.lower() not in self.stop_words
        ]
        
        # Remove duplicates and sort
        return sorted(list(set(filtered_keywords)))
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities using spaCy."""
        entities = set()
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Focus on important entity types
                if ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG', 'WORK_OF_ART', 'EVENT']:
                    # Clean entity text
                    clean_entity = self._clean_text(ent.text)
                    if len(clean_entity) > 1:
                        entities.add(clean_entity)
            
        except Exception as e:
            logger.debug(f"Error extracting entities: {e}")
        
        return entities
    
    def _extract_important_words(self, text: str) -> Set[str]:
        """Extract important words using POS tagging."""
        important_words = set()
        
        try:
            if self.nlp:
                doc = self.nlp(text)
                
                for token in doc:
                    # Focus on content words
                    if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
                        not token.is_stop and 
                        not token.is_punct and
                        len(token.text) > 2):
                        important_words.add(token.lemma_.lower())
            else:
                # Fallback: simple word extraction
                words = text.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        important_words.add(word.lower())
        
        except Exception as e:
            logger.debug(f"Error extracting important words: {e}")
        
        return important_words
    
    def _extract_travel_terms(self, text: str) -> Set[str]:
        """Extract travel and location specific terms."""
        travel_terms = set()
        
        # Common travel-related terms
        travel_keywords = [
            'visit', 'travel', 'trip', 'journey', 'place', 'places', 'location', 'locations',
            'city', 'cities', 'country', 'countries', 'region', 'regions', 'area', 'areas',
            'weather', 'climate', 'temperature', 'rain', 'sunny', 'cloudy', 'snow',
            'hotel', 'restaurant', 'museum', 'attraction', 'attractions', 'sightseeing',
            'italy', 'france', 'spain', 'germany', 'england', 'rome', 'paris', 'london',
            'venice', 'florence', 'milan', 'naples', 'turin', 'bologna', 'genoa',
            'mark', 'twain', 'author', 'writer', 'book', 'books', 'literature'
        ]
        
        text_lower = text.lower()
        for term in travel_keywords:
            if term in text_lower:
                travel_terms.add(term)
        
        return travel_terms
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning as fallback."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def get_embedding_query(self, original_query: str) -> str:
        """
        Get the processed query optimized for embedding generation.
        
        Args:
            original_query: Original user query
            
        Returns:
            Processed query for embedding
        """
        keyword_query, _ = self.preprocess_query(original_query)
        return keyword_query if keyword_query else original_query
