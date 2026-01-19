#!/usr/bin/env python3
"""
Intent Confidence Mapper
Validates primary intents and enrichments using algorithmic confidence scoring
No LLM usage - pure code-based validation
"""

import re
from typing import Dict, List, Set, Tuple, Any
from difflib import SequenceMatcher


class IntentConfidenceMapper:
    """Calculate confidence scores for intent classification without using LLM"""

    def __init__(
        self,
        intent_categories: Dict,
        enrichment_rules: Dict,
        keyword_weight: float = 0.4,
        similarity_weight: float = 0.6
    ):
        """
        Initialize confidence mapper

        Args:
            intent_categories: Loaded intent_categories.yaml
            enrichment_rules: Loaded enrichment_rules.yaml
            keyword_weight: Weight for keyword matching (default: 0.4)
            similarity_weight: Weight for example similarity (default: 0.6)
        """
        self.intent_categories = intent_categories
        self.enrichment_rules = enrichment_rules
        self.keyword_weight = keyword_weight
        self.similarity_weight = similarity_weight

        # Build lookup structures
        self.all_valid_intents = self._build_valid_intents()
        self.intent_keywords = self._build_intent_keywords()
        self.intent_examples = self._build_intent_examples()

    def _build_valid_intents(self) -> Set[str]:
        """Build set of all valid intent names"""
        valid_intents = set()
        for category, category_data in self.intent_categories.items():
            if 'intents' in category_data:
                valid_intents.update(category_data['intents'].keys())
        return valid_intents

    def _build_intent_keywords(self) -> Dict[str, List[str]]:
        """Extract keywords from intent descriptions and examples"""
        intent_keywords = {}

        for category, category_data in self.intent_categories.items():
            if 'intents' in category_data:
                for intent_name, intent_data in category_data['intents'].items():
                    keywords = []

                    # Extract from description
                    description = intent_data.get('description', '')
                    keywords.extend(self._extract_keywords(description))

                    # Extract from examples
                    examples = intent_data.get('examples', [])
                    for example in examples:
                        keywords.extend(self._extract_keywords(example))

                    intent_keywords[intent_name] = list(set(keywords))

        return intent_keywords

    def _build_intent_examples(self) -> Dict[str, List[str]]:
        """Build mapping of intents to their examples"""
        intent_examples = {}

        for category, category_data in self.intent_categories.items():
            if 'intents' in category_data:
                for intent_name, intent_data in category_data['intents'].items():
                    intent_examples[intent_name] = intent_data.get('examples', [])

        return intent_examples

    def _simple_stem(self, word: str) -> str:
        """
        Simple stemming for common English suffixes
        Not as sophisticated as Porter/Snowball, but good enough without dependencies
        """
        # Remove common suffixes
        if len(word) <= 3:
            return word

        original = word

        # Progressive/gerund (-ing) - do this FIRST before plural
        if word.endswith('ing') and len(word) > 5:
            word = word[:-3]
            # Handle double consonants (running -> run, not runn)
            if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in 'aeiou':
                word = word[:-1]

        # Past tense (-ed)
        if word.endswith('ed') and len(word) > 4:
            word = word[:-2]
            # Handle double consonants
            if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in 'aeiou':
                word = word[:-1]

        # Plural forms
        if word.endswith('ies') and len(word) > 4:
            word = word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            word = word[:-2]  # services -> servic, processes -> process
        elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
            word = word[:-1]

        # Additional normalization: ce -> c (service -> servic to match services)
        if word.endswith('ce') and len(word) > 4:
            word = word[:-1]

        return word

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text with stemming"""
        # Convert to lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)

        # Split into words
        words = text.split()

        # Filter out common stop words (reduced list - keep question words for context)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            # NOTE: Removed 'what', 'which', 'who', 'when', 'where', 'why', 'how'
            # These are important for intent context!
        }

        # Apply stemming to each word
        keywords = []
        for w in words:
            if w not in stop_words and len(w) > 2:
                stemmed = self._simple_stem(w)
                keywords.append(stemmed)

        return keywords

    def calculate_primary_intent_confidence(
        self,
        query: str,
        intent: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate confidence score for a primary intent

        Args:
            query: User's query
            intent: Detected primary intent

        Returns:
            Tuple of (confidence_score, details_dict)
        """
        details = {
            'intent': intent,
            'is_valid': False,
            'keyword_match_score': 0.0,
            'example_similarity_score': 0.0,
            'final_score': 0.0,
            'matched_keywords': [],
            'reasoning': []
        }

        # Check if intent is valid
        if intent not in self.all_valid_intents:
            details['reasoning'].append(f"Intent '{intent}' not found in configuration")
            return 0.0, details

        details['is_valid'] = True
        details['reasoning'].append(f"Intent '{intent}' is valid")

        # Extract query keywords
        query_keywords = set(self._extract_keywords(query))

        # Calculate keyword match score using F1-score approach
        intent_keywords = set(self.intent_keywords.get(intent, []))
        if intent_keywords and query_keywords:
            matched_keywords = query_keywords.intersection(intent_keywords)

            # Precision: what % of query keywords matched?
            precision = len(matched_keywords) / len(query_keywords) if query_keywords else 0

            # Recall: what % of intent keywords were found?
            recall = len(matched_keywords) / len(intent_keywords) if intent_keywords else 0

            # F1-score: harmonic mean of precision and recall
            if precision + recall > 0:
                keyword_match_score = 2 * (precision * recall) / (precision + recall)
            else:
                keyword_match_score = 0.0

            details['keyword_match_score'] = keyword_match_score
            details['keyword_precision'] = precision
            details['keyword_recall'] = recall
            details['matched_keywords'] = list(matched_keywords)
            details['query_keywords'] = list(query_keywords)
            details['intent_keywords'] = list(intent_keywords)

            if matched_keywords:
                details['reasoning'].append(
                    f"Matched {len(matched_keywords)}/{len(query_keywords)} query keywords: {', '.join(matched_keywords)}"
                )
                details['reasoning'].append(
                    f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {keyword_match_score:.2f}"
                )

        # Calculate example similarity score
        examples = self.intent_examples.get(intent, [])
        if examples:
            max_similarity = 0.0
            best_example = None

            for example in examples:
                similarity = self._calculate_text_similarity(query, example)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_example = example

            details['example_similarity_score'] = max_similarity
            if max_similarity > 0.3:
                details['reasoning'].append(
                    f"Query similar to example: '{best_example}' (score: {max_similarity:.2f})"
                )

        # Calculate final confidence score using configurable weights
        final_score = (
            self.keyword_weight * details['keyword_match_score'] +
            self.similarity_weight * details['example_similarity_score']
        )

        details['final_score'] = final_score
        details['weights_used'] = {
            'keyword': self.keyword_weight,
            'similarity': self.similarity_weight
        }

        # Determine confidence level
        if final_score >= 0.7:
            details['reasoning'].append("HIGH confidence - strong match")
        elif final_score >= 0.4:
            details['reasoning'].append("MEDIUM confidence - acceptable match")
        elif final_score >= 0.2:
            details['reasoning'].append("LOW confidence - weak match")
        else:
            details['reasoning'].append("VERY LOW confidence - questionable match")

        return final_score, details

    def calculate_enrichment_confidence(
        self,
        primary_intent: str,
        enrichment_intent: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate confidence that an enrichment intent is legitimate

        Args:
            primary_intent: The primary intent
            enrichment_intent: The enrichment intent to validate

        Returns:
            Tuple of (confidence_score, details_dict)
        """
        details = {
            'primary_intent': primary_intent,
            'enrichment_intent': enrichment_intent,
            'is_valid_enrichment': False,
            'is_valid_intent': False,
            'enrichment_depth': 0,
            'final_score': 0.0,
            'reasoning': []
        }

        # Check if enrichment intent exists in configuration
        if enrichment_intent not in self.all_valid_intents:
            details['reasoning'].append(
                f"Enrichment '{enrichment_intent}' not found in configuration"
            )
            return 0.0, details

        details['is_valid_intent'] = True

        # Check if this enrichment is directly linked to primary intent
        direct_enrichments = self.enrichment_rules.get(primary_intent, [])

        if enrichment_intent in direct_enrichments:
            details['is_valid_enrichment'] = True
            details['enrichment_depth'] = 1
            details['final_score'] = 1.0
            details['reasoning'].append(
                f"Direct enrichment rule: {primary_intent} → {enrichment_intent}"
            )
            return 1.0, details

        # Check if enrichment is secondary (enrichment of enrichment)
        for direct_enrichment in direct_enrichments:
            secondary_enrichments = self.enrichment_rules.get(direct_enrichment, [])
            if enrichment_intent in secondary_enrichments:
                details['is_valid_enrichment'] = True
                details['enrichment_depth'] = 2
                details['final_score'] = 0.8
                details['reasoning'].append(
                    f"Secondary enrichment: {primary_intent} → {direct_enrichment} → {enrichment_intent}"
                )
                return 0.8, details

        # Not a valid enrichment for this primary intent
        details['reasoning'].append(
            f"No enrichment rule found: {primary_intent} → {enrichment_intent}"
        )
        return 0.0, details

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using SequenceMatcher

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()

        # Also check keyword overlap
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))

        if keywords1 and keywords2:
            keyword_overlap = len(keywords1.intersection(keywords2)) / max(len(keywords1), len(keywords2))
            # Combine both scores (70% sequence match, 30% keyword overlap)
            similarity = 0.7 * similarity + 0.3 * keyword_overlap

        return similarity

    def validate_classification_result(
        self,
        query: str,
        primary_intents: List[str],
        enriched_intents: List[str]
    ) -> Dict[str, Any]:
        """
        Validate entire classification result with confidence scores

        Args:
            query: User's original query
            primary_intents: List of primary intents from LLM
            enriched_intents: List of all enriched intents

        Returns:
            Validation result with confidence scores
        """
        result = {
            'query': query,
            'primary_intents': {},
            'enrichments': {},
            'overall_confidence': 0.0,
            'recommendations': []
        }

        # Validate primary intents
        primary_scores = []
        for intent in primary_intents:
            score, details = self.calculate_primary_intent_confidence(query, intent)
            result['primary_intents'][intent] = {
                'confidence_score': score,
                'details': details
            }
            primary_scores.append(score)

            # Add recommendations for low confidence
            if score < 0.4:
                result['recommendations'].append(
                    f"Low confidence for primary intent '{intent}' (score: {score:.2f}). Consider reviewing."
                )

        # Validate enrichments
        enrichment_scores = []
        enrichment_intents = [e for e in enriched_intents if e not in primary_intents]

        for enrichment in enrichment_intents:
            # Find which primary intent(s) this enrichment came from
            enrichment_sources = []
            for primary in primary_intents:
                score, details = self.calculate_enrichment_confidence(primary, enrichment)
                if score > 0:
                    enrichment_sources.append({
                        'primary': primary,
                        'score': score,
                        'details': details
                    })
                    enrichment_scores.append(score)

            if enrichment_sources:
                result['enrichments'][enrichment] = {
                    'sources': enrichment_sources,
                    'max_confidence': max(s['score'] for s in enrichment_sources)
                }
            else:
                # Enrichment not linked to any primary intent
                result['enrichments'][enrichment] = {
                    'sources': [],
                    'max_confidence': 0.0
                }
                result['recommendations'].append(
                    f"Enrichment '{enrichment}' has no valid link to primary intents. Possible error."
                )

        # Calculate overall confidence
        # IMPORTANT: Primary intent confidence is what matters most!
        # Enrichments being valid doesn't make bad primary intents good
        if primary_scores:
            primary_confidence = sum(primary_scores) / len(primary_scores)
        else:
            primary_confidence = 0.0

        if enrichment_scores:
            enrichment_confidence = sum(enrichment_scores) / len(enrichment_scores)
        else:
            enrichment_confidence = 1.0  # No enrichments = nothing wrong

        # Overall confidence is primarily based on primary intents (80% weight)
        # Enrichments contribute less (20% weight) - they're just validation
        result['overall_confidence'] = 0.8 * primary_confidence + 0.2 * enrichment_confidence
        result['primary_confidence'] = primary_confidence
        result['enrichment_confidence'] = enrichment_confidence

        # Overall recommendations based on PRIMARY confidence
        if primary_confidence >= 0.7:
            result['recommendations'].insert(0, "Overall confidence is HIGH. Classification looks good.")
        elif primary_confidence >= 0.4:
            result['recommendations'].insert(0, "Overall confidence is MEDIUM. Review recommended.")
        else:
            result['recommendations'].insert(0, "Overall confidence is LOW. Manual review required.")

        return result

    def get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level"""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"


def calculate_token_consumption(
    system_prompt: str,
    user_query: str,
    response: str,
    model_id: str = "claude-3-5-sonnet"
) -> Dict[str, Any]:
    """
    Calculate approximate token consumption for LLM call

    Note: This is an approximation. Actual token count may vary.
    Claude models use ~4 characters per token on average for English text.

    Args:
        system_prompt: The system prompt sent
        user_query: User's query
        response: LLM response
        model_id: Model identifier

    Returns:
        Dict with token counts and estimates
    """
    # Approximate tokens (4 chars per token for English)
    CHARS_PER_TOKEN = 4

    # Calculate character counts
    system_chars = len(system_prompt)
    query_chars = len(user_query)
    response_chars = len(response)

    # Estimate tokens
    system_tokens = system_chars / CHARS_PER_TOKEN
    query_tokens = query_chars / CHARS_PER_TOKEN
    response_tokens = response_chars / CHARS_PER_TOKEN

    input_tokens = system_tokens + query_tokens
    output_tokens = response_tokens
    total_tokens = input_tokens + output_tokens

    # Pricing (as of 2024 - update as needed)
    # Claude 3.5 Sonnet pricing on Bedrock
    pricing = {
        'claude-3-5-sonnet': {
            'input': 0.003,   # per 1K tokens
            'output': 0.015   # per 1K tokens
        },
        'claude-3-haiku': {
            'input': 0.00025,
            'output': 0.00125
        }
    }

    # Get pricing for model
    model_pricing = pricing.get(
        'claude-3-5-sonnet',  # default
        pricing['claude-3-5-sonnet']
    )

    # Calculate cost
    input_cost = (input_tokens / 1000) * model_pricing['input']
    output_cost = (output_tokens / 1000) * model_pricing['output']
    total_cost = input_cost + output_cost

    return {
        'input_tokens': int(input_tokens),
        'output_tokens': int(output_tokens),
        'total_tokens': int(total_tokens),
        'breakdown': {
            'system_prompt_tokens': int(system_tokens),
            'user_query_tokens': int(query_tokens),
            'response_tokens': int(response_tokens)
        },
        'cost_usd': {
            'input': round(input_cost, 6),
            'output': round(output_cost, 6),
            'total': round(total_cost, 6)
        },
        'model': model_id,
        'note': 'Approximate calculation using ~4 chars/token. Actual may vary.'
    }
