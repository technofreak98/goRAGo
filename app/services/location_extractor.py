"""Location extraction service using spaCy NER for identifying places in text."""

import logging
import re
from typing import List, Set, Dict, Any, Optional
import spacy
from app.models import SearchResult

logger = logging.getLogger(__name__)


class LocationExtractor:
    """Service for extracting location entities from text using spaCy NER."""
    
    def __init__(self):
        """Initialize location extractor with spaCy model."""
        self.nlp = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize spaCy model for NER."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Location extractor initialized with spaCy model")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        except Exception as e:
            logger.error(f"Error initializing location extractor: {e}")
            self.nlp = None
    
    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location entities from text.
        
        Args:
            text: Input text to extract locations from
            
        Returns:
            List of unique location names
        """
        try:
            if not self.nlp:
                logger.warning("spaCy model not available, using fallback extraction")
                return self._fallback_location_extraction(text)
            
            doc = self.nlp(text)
            locations = set()
            
            # Extract named entities
            for ent in doc.ents:
                if self._is_location_entity(ent):
                    clean_location = self._clean_location_name(ent.text)
                    if clean_location:
                        locations.add(clean_location)
            
            # Extract location patterns using regex
            regex_locations = self._extract_location_patterns(text)
            locations.update(regex_locations)
            
            # Filter and validate locations
            filtered_locations = self._filter_locations(list(locations))
            
            logger.debug(f"Extracted locations from text: {filtered_locations}")
            return filtered_locations
            
        except Exception as e:
            logger.error(f"Error extracting locations: {e}")
            return []
    
    def extract_from_chunks(self, chunks: List[SearchResult]) -> List[str]:
        """
        Extract locations from a list of search result chunks.
        
        Args:
            chunks: List of SearchResult objects
            
        Returns:
            List of unique location names found in chunks
        """
        try:
            all_locations = set()
            
            for chunk in chunks:
                # Extract from main text
                chunk_locations = self.extract_locations(chunk.text)
                all_locations.update(chunk_locations)
                
                # Extract from parent window if available
                if chunk.parent_window:
                    parent_locations = self.extract_locations(chunk.parent_window)
                    all_locations.update(parent_locations)
                
                # Extract from context if available
                if chunk.context:
                    context_locations = self.extract_locations(chunk.context)
                    all_locations.update(context_locations)
            
            # Filter and deduplicate
            filtered_locations = self._filter_locations(list(all_locations))
            
            logger.info(f"Extracted {len(filtered_locations)} unique locations from {len(chunks)} chunks")
            return filtered_locations
            
        except Exception as e:
            logger.error(f"Error extracting locations from chunks: {e}")
            return []
    
    def _is_location_entity(self, entity) -> bool:
        """Check if a spaCy entity is a location."""
        location_labels = {
            'GPE',  # Geopolitical entity (countries, cities, states)
            'LOC',  # Location (mountains, bodies of water, etc.)
            'FAC',  # Facility (buildings, airports, highways, etc.)
            'ORG'   # Organization (sometimes includes place names)
        }
        
        return entity.label_ in location_labels
    
    def _clean_location_name(self, location: str) -> str:
        """Clean and normalize location name."""
        # Remove extra whitespace
        location = location.strip()
        
        # Remove common prefixes/suffixes
        location = re.sub(r'^(the|a|an)\s+', '', location, flags=re.IGNORECASE)
        location = re.sub(r'\s+(city|town|village|place)$', '', location, flags=re.IGNORECASE)
        
        # Remove special characters but keep spaces and hyphens
        location = re.sub(r'[^\w\s\-]', '', location)
        
        # Normalize whitespace
        location = re.sub(r'\s+', ' ', location)
        
        # Filter out very short or generic terms
        if len(location) < 2 or location.lower() in {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at'}:
            return ""
        
        return location
    
    def _extract_location_patterns(self, text: str) -> List[str]:
        """Extract location patterns using regex."""
        locations = set()
        
        # Common location patterns
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Italy|France|Spain|Germany|England|UK|USA|America)\b',
            r'\b(?:Rome|Venice|Florence|Milan|Naples|Turin|Bologna|Genoa|Pisa|Verona|Padua|Ravenna)\b',
            r'\b(?:Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier)\b',
            r'\b(?:London|Manchester|Birmingham|Liverpool|Leeds|Sheffield|Bristol|Newcastle)\b',
            r'\b(?:Madrid|Barcelona|Valencia|Seville|Zaragoza|Málaga|Murcia|Palma)\b',
            r'\b(?:Berlin|Hamburg|Munich|Cologne|Frankfurt|Stuttgart|Düsseldorf|Dortmund)\b',
            r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego)\b',
            r'\b(?:Tokyo|Osaka|Kyoto|Yokohama|Nagoya|Sapporo|Fukuoka|Kobe)\b',
            r'\b(?:Beijing|Shanghai|Guangzhou|Shenzhen|Tianjin|Wuhan|Dongguan|Chongqing)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_location = self._clean_location_name(match)
                if clean_location:
                    locations.add(clean_location)
        
        return list(locations)
    
    def _filter_locations(self, locations: List[str]) -> List[str]:
        """Filter and validate extracted locations."""
        if not locations:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        filtered = []
        
        for location in locations:
            location_lower = location.lower()
            if location_lower not in seen and len(location) > 1:
                seen.add(location_lower)
                filtered.append(location)
        
        # Sort by length (longer names first) and then alphabetically
        filtered.sort(key=lambda x: (-len(x), x.lower()))
        
        return filtered
    
    def _fallback_location_extraction(self, text: str) -> List[str]:
        """Fallback location extraction without spaCy."""
        locations = set()
        
        # Use regex patterns as fallback
        regex_locations = self._extract_location_patterns(text)
        locations.update(regex_locations)
        
        # Simple capitalization-based extraction
        words = text.split()
        for i, word in enumerate(words):
            # Look for capitalized words that might be locations
            if (word[0].isupper() and 
                len(word) > 2 and 
                word.isalpha() and
                not word.lower() in {'The', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}):
                
                # Check if next word is also capitalized (compound location)
                if (i + 1 < len(words) and 
                    words[i + 1][0].isupper() and 
                    words[i + 1].isalpha()):
                    compound_location = f"{word} {words[i + 1]}"
                    locations.add(compound_location)
                else:
                    locations.add(word)
        
        return self._filter_locations(list(locations))
    
    def get_location_context(self, locations: List[str], chunks: List[SearchResult]) -> Dict[str, List[str]]:
        """
        Get context for each location from the chunks.
        
        Args:
            locations: List of location names
            chunks: List of SearchResult objects
            
        Returns:
            Dictionary mapping locations to relevant chunk texts
        """
        location_context = {location: [] for location in locations}
        
        try:
            for chunk in chunks:
                chunk_text = f"{chunk.text} {chunk.parent_window or ''} {chunk.context or ''}"
                chunk_text_lower = chunk_text.lower()
                
                for location in locations:
                    if location.lower() in chunk_text_lower:
                        # Add chunk info with relevance
                        context_info = {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            "relevance_score": chunk.relevance_score or 0.0,
                            "document_id": chunk.document_id
                        }
                        location_context[location].append(context_info)
            
            # Sort by relevance score
            for location in location_context:
                location_context[location].sort(
                    key=lambda x: x["relevance_score"], 
                    reverse=True
                )
            
            return location_context
            
        except Exception as e:
            logger.error(f"Error getting location context: {e}")
            return {location: [] for location in locations}
    
    def validate_locations(self, locations: List[str]) -> List[str]:
        """
        Validate and clean a list of location names.
        
        Args:
            locations: List of location names to validate
            
        Returns:
            List of validated location names
        """
        if not locations:
            return []
        
        validated = []
        
        for location in locations:
            clean_location = self._clean_location_name(location)
            if clean_location and len(clean_location) > 1:
                validated.append(clean_location)
        
        return self._filter_locations(validated)
