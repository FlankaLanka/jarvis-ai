"""
Jarvis Voice Assistant - Zero-Hallucination Guardrails

Verification layer to ensure responses are accurate and grounded in data.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ConfidenceLevel(Enum):
    """Confidence levels for response verification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


@dataclass
class VerificationResult:
    """Result of response verification."""
    is_valid: bool
    confidence: ConfidenceLevel
    verified_claims: List[str]
    unverified_claims: List[str]
    suggested_response: Optional[str] = None


class GuardrailsService:
    """
    Service for verifying LLM responses and preventing hallucinations.
    
    Strategies:
    1. Cross-reference with verified data sources (GitHub, APIs)
    2. Detect speculative language and flag uncertain claims
    3. Provide confidence scores
    4. Suggest safe fallback responses for unverified claims
    """
    
    def __init__(self):
        # Keywords that indicate speculation or uncertainty
        self._uncertainty_markers = [
            "might", "maybe", "possibly", "could be", "probably",
            "i think", "i believe", "it seems", "perhaps", "likely",
            "not sure", "uncertain", "guessing"
        ]
        
        # Keywords that indicate factual claims
        self._factual_markers = [
            "is", "are", "was", "were", "will", "has", "have",
            "the", "there are", "there is", "according to"
        ]
    
    def verify_response(
        self,
        response: str,
        query: str,
        verified_data: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify an LLM response for accuracy.
        
        Args:
            response: The LLM's response to verify
            query: The original query
            verified_data: Data from verified sources (GitHub, APIs)
            
        Returns:
            Verification result with confidence and claims
        """
        response_lower = response.lower()
        
        # Check for uncertainty markers
        uncertainty_count = sum(
            1 for marker in self._uncertainty_markers
            if marker in response_lower
        )
        
        # If we have verified data, check claims against it
        verified_claims = []
        unverified_claims = []
        
        if verified_data:
            # Simple claim verification (can be made more sophisticated)
            for key, value in verified_data.items():
                value_str = str(value).lower()
                if value_str in response_lower:
                    verified_claims.append(f"{key}: {value}")
        
        # Determine confidence level
        if uncertainty_count > 2:
            confidence = ConfidenceLevel.LOW
        elif uncertainty_count > 0:
            confidence = ConfidenceLevel.MEDIUM
        elif verified_claims:
            confidence = ConfidenceLevel.HIGH
        else:
            confidence = ConfidenceLevel.MEDIUM
        
        # Check if response is valid
        is_valid = confidence != ConfidenceLevel.LOW or len(verified_claims) > 0
        
        # Generate suggested response if needed
        suggested_response = None
        if not is_valid:
            suggested_response = self._generate_safe_response(query)
        
        result = VerificationResult(
            is_valid=is_valid,
            confidence=confidence,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            suggested_response=suggested_response
        )
        
        logger.debug(f"Verification result: valid={is_valid}, confidence={confidence.value}")
        return result
    
    def _generate_safe_response(self, query: str) -> str:
        """
        Generate a safe fallback response when verification fails.
        
        Args:
            query: The original query
            
        Returns:
            A safe fallback response
        """
        return (
            "I don't have verified information to answer that question accurately. "
            "Would you like me to help you find a reliable source for this information?"
        )
    
    def add_confidence_disclaimer(
        self,
        response: str,
        confidence: ConfidenceLevel
    ) -> str:
        """
        Add a confidence disclaimer to a response if needed.
        
        Args:
            response: The response to modify
            confidence: The confidence level
            
        Returns:
            Response with disclaimer if needed
        """
        if confidence == ConfidenceLevel.LOW:
            return f"Note: I'm not fully confident about this. {response}"
        elif confidence == ConfidenceLevel.UNVERIFIED:
            return f"I couldn't verify this information: {response}"
        return response
    
    def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from a response.
        
        Args:
            response: The response to analyze
            
        Returns:
            List of extracted claims
        """
        # Simple sentence-based claim extraction
        # Can be made more sophisticated with NLP
        sentences = response.replace("!", ".").replace("?", ".").split(".")
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            
            # Check if sentence contains factual markers
            if any(marker in sentence_lower for marker in self._factual_markers):
                # Skip if it's a question or uncertain
                if not any(marker in sentence_lower for marker in self._uncertainty_markers):
                    claims.append(sentence)
        
        return claims
    
    def validate_against_source(
        self,
        claim: str,
        source_data: Dict[str, Any]
    ) -> bool:
        """
        Validate a specific claim against source data.
        
        Args:
            claim: The claim to validate
            source_data: Data from a verified source
            
        Returns:
            True if claim can be verified, False otherwise
        """
        claim_lower = claim.lower()
        
        for key, value in source_data.items():
            value_str = str(value).lower()
            key_lower = key.lower()
            
            # Check if both the key and value appear in the claim
            if key_lower in claim_lower and value_str in claim_lower:
                return True
        
        return False


# Global guardrails service instance
guardrails_service = GuardrailsService()

