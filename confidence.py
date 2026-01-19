"""Confidence scoring system for intent classification"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ConfidenceScorer:
    """Score and manage confidence for intent classification"""

    def __init__(
        self,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.7,
        low_threshold: float = 0.5
    ):
        """
        Initialize confidence scorer

        Args:
            high_threshold: Threshold for high confidence
            medium_threshold: Threshold for medium confidence
            low_threshold: Threshold for low confidence
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

        logger.info(
            f"Confidence scorer initialized: "
            f"high={high_threshold}, medium={medium_threshold}, low={low_threshold}"
        )

    def score(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence score for classification result

        This is a placeholder implementation. In production, you would:
        1. Use embedding similarity between question and intent examples
        2. Check if LLM returned explanatory text (lower confidence)
        3. Use model's own confidence scores if available
        4. Track historical accuracy per intent

        Args:
            classification_result: Result from intent classifier

        Returns:
            Enhanced result with confidence scores
        """
        intent = classification_result.get("intent", "")
        question = classification_result.get("question", "")

        # Placeholder: Calculate basic confidence
        # In production, this would use more sophisticated methods
        confidence_score = self._calculate_confidence(intent, question)

        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)

        # Add confidence info to result
        result = {
            **classification_result,
            "confidence": confidence_score,
            "confidence_level": confidence_level.value,
            "confidence_details": {
                "score": confidence_score,
                "level": confidence_level.value,
                "thresholds": {
                    "high": self.high_threshold,
                    "medium": self.medium_threshold,
                    "low": self.low_threshold
                }
            }
        }

        logger.info(
            f"Confidence scoring: {intent} -> {confidence_score:.2f} ({confidence_level.value})"
        )

        return result

    def _calculate_confidence(self, intent: str, question: str) -> float:
        """
        Calculate confidence score

        Placeholder implementation - would be enhanced with:
        - Embedding similarity
        - Historical accuracy
        - Model confidence scores
        - Question clarity metrics

        Args:
            intent: Classified intent
            question: Original question

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Placeholder: Simple heuristics
        score = 0.8  # Base score

        # Adjust based on question length (too short or too long = less confident)
        question_length = len(question.split())
        if question_length < 3:
            score -= 0.15
        elif question_length > 30:
            score -= 0.1

        # Adjust based on question marks (clear questions = higher confidence)
        if "?" in question:
            score += 0.05

        # Check for vague terms (lower confidence)
        vague_terms = ["something", "anything", "whatever", "stuff", "things", "maybe"]
        if any(term in question.lower() for term in vague_terms):
            score -= 0.2

        # Check for specific service names (higher confidence)
        if any(indicator in question.lower() for indicator in ["-api", "service", "app"]):
            score += 0.05

        # Ensure score is in valid range
        return max(0.0, min(1.0, score))

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """
        Convert score to confidence level

        Args:
            score: Confidence score

        Returns:
            Confidence level enum
        """
        if score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif score >= self.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def should_request_clarification(self, confidence_score: float) -> bool:
        """
        Determine if clarification should be requested from user

        Args:
            confidence_score: Confidence score

        Returns:
            True if clarification recommended
        """
        return confidence_score < self.low_threshold

    def get_fallback_intent(
        self,
        confidence_score: float,
        category: str
    ) -> Optional[str]:
        """
        Get fallback intent if confidence is too low

        Args:
            confidence_score: Confidence score
            category: Intent category

        Returns:
            Fallback intent name or None
        """
        if confidence_score >= self.low_threshold:
            return None

        # Map category to safe fallback intents
        fallbacks = {
            "STATE": "CURRENT_HEALTH",
            "TREND": "UNDERCURRENTS_TREND",
            "PATTERN": "RECURRING_INCIDENT",
            "CAUSE": "ROOT_CAUSE_SINGLE",
            "IMPACT": "BLAST_RADIUS",
            "ACTION": "MITIGATION_STEPS",
            "PREDICT": "RISK_PREDICTION",
            "OPTIMIZE": "PERFORMANCE_BOTTLENECK",
            "EVIDENCE": "EVIDENCE_SUMMARY"
        }

        return fallbacks.get(category)

    def suggest_improvements(self, confidence_score: float) -> List[str]:
        """
        Suggest how to improve confidence

        Args:
            confidence_score: Current confidence score

        Returns:
            List of suggestions
        """
        suggestions = []

        if confidence_score < self.medium_threshold:
            suggestions.append(
                "Try to be more specific in your question"
            )
            suggestions.append(
                "Include service names or specific metrics if applicable"
            )

        if confidence_score < self.low_threshold:
            suggestions.append(
                "Rephrase the question with clearer intent"
            )
            suggestions.append(
                "Break down complex questions into simpler ones"
            )

        return suggestions

    def get_confidence_stats(
        self,
        classification_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get confidence statistics from classification history

        Args:
            classification_history: List of past classifications with confidence

        Returns:
            Dict with confidence statistics
        """
        if not classification_history:
            return {
                "total_classifications": 0,
                "average_confidence": 0.0,
                "distribution": {}
            }

        total = len(classification_history)
        total_confidence = sum(
            c.get("confidence", 0.0) for c in classification_history
        )
        avg_confidence = total_confidence / total

        # Distribution by level
        distribution = {level.value: 0 for level in ConfidenceLevel}
        for classification in classification_history:
            score = classification.get("confidence", 0.0)
            level = self._get_confidence_level(score)
            distribution[level.value] += 1

        # Convert counts to percentages
        distribution = {
            level: (count / total) * 100
            for level, count in distribution.items()
        }

        return {
            "total_classifications": total,
            "average_confidence": avg_confidence,
            "distribution": distribution,
            "high_confidence_rate": distribution.get(ConfidenceLevel.HIGH.value, 0),
            "clarification_rate": sum(
                1 for c in classification_history
                if self.should_request_clarification(c.get("confidence", 0.0))
            ) / total * 100
        }

    def adjust_thresholds(
        self,
        high: Optional[float] = None,
        medium: Optional[float] = None,
        low: Optional[float] = None
    ):
        """
        Dynamically adjust confidence thresholds

        Args:
            high: New high threshold
            medium: New medium threshold
            low: New low threshold
        """
        if high is not None:
            self.high_threshold = high
        if medium is not None:
            self.medium_threshold = medium
        if low is not None:
            self.low_threshold = low

        logger.info(
            f"Thresholds adjusted: high={self.high_threshold}, "
            f"medium={self.medium_threshold}, low={self.low_threshold}"
        )