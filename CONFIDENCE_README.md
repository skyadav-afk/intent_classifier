# Intent Confidence System

## Overview

The confidence system validates intent classifications **without using LLMs**. It provides algorithmic confidence scores for both primary intents (from LLM) and enriched intents (from rules).

## Quick Start

```python
from intent_classifier import IntentClassifier

# Initialize (confidence mapper auto-loaded)
classifier = IntentClassifier()

# Classify with confidence
result = classifier.classify("Why is payment-api failing?")

# Access confidence data
print(f"Overall: {result['confidence_validation']['overall_confidence']:.2f}")
print(f"Primary: {result['confidence_validation']['primary_confidence']:.2f}")
print(f"Tokens: {result['token_consumption']['total_tokens']}")

# Print full result
classifier.print_result(result)
```

## How It Works

### 1. Primary Intent Confidence (0.0 to 1.0)

Validates that LLM's detected intent matches the query using:

**Keyword Matching (40% weight):**
- Extracts keywords with stemming (services→servic, failing→fail)
- Calculates F1-score (harmonic mean of precision & recall)
- Precision: What % of query keywords matched?
- Recall: What % of intent keywords found?

**Example Similarity (60% weight):**
- Compares query to intent examples using SequenceMatcher
- Also checks keyword overlap
- Takes best matching example

**Final Score:** `0.4 * keyword_f1 + 0.6 * example_similarity`

**Confidence Levels:**
- HIGH (≥0.7): Strong match
- MEDIUM (≥0.4): Acceptable match
- LOW (≥0.2): Weak match - review needed
- VERY_LOW (<0.2): Questionable - likely wrong

### 2. Enrichment Validation (0.0 or 1.0)

Checks if enrichment exists in `enrichment_rules.yaml`:
- **1.0**: Direct enrichment rule exists
- **0.8**: Secondary enrichment (2 hops)
- **0.0**: No rule found (error)

### 3. Overall Confidence

```
overall = 0.8 * primary_confidence + 0.2 * enrichment_confidence
```

**Important:** Primary intent confidence matters most! Valid enrichments don't make bad primary intents good.

### 4. Token Consumption

Tracks LLM usage per call:
- Approximates tokens (~4 chars/token)
- Calculates costs (Claude 3.5 Sonnet pricing)
- Shows input/output breakdown

## Configuration

```python
from intent_confidence import IntentConfidenceMapper

# Custom weights
mapper = IntentConfidenceMapper(
    intent_categories,
    enrichment_rules,
    keyword_weight=0.5,      # Default: 0.4
    similarity_weight=0.5    # Default: 0.6
)
```

## Testing

```bash
# Offline tests (no AWS needed)
python3 test_confidence_offline.py

# Live tests (requires AWS credentials)
python3 test_confidence.py

# Test specific query
python3 test_specific_query.py
```

## Result Format

```python
{
    "confidence_validation": {
        "overall_confidence": 0.38,
        "primary_confidence": 0.22,      # What matters!
        "enrichment_confidence": 1.0,

        "primary_intents": {
            "SERVICE_HEALTH": {
                "confidence_score": 0.29,
                "details": {
                    "keyword_match_score": 0.13,
                    "keyword_precision": 0.20,
                    "keyword_recall": 0.09,
                    "example_similarity_score": 0.40,
                    "matched_keywords": ["servic"],
                    "reasoning": [...]
                }
            }
        },

        "enrichments": {
            "UNDERCURRENTS_TREND": {
                "max_confidence": 1.0,
                "sources": [{
                    "primary": "SERVICE_HEALTH",
                    "score": 1.0
                }]
            }
        },

        "recommendations": [
            "Overall confidence is LOW. Manual review required.",
            "Low confidence for primary intent 'SERVICE_HEALTH' (score: 0.29)."
        ]
    },

    "token_consumption": {
        "input_tokens": 940,
        "output_tokens": 12,
        "total_tokens": 952,
        "cost_usd": {
            "input": 0.002821,
            "output": 0.000180,
            "total": 0.003001
        }
    }
}
```

## Recent Fixes (Critical Issues Resolved)

### ❌ Problem 1: Wrong Formula
**Before:** `matched / intent_keywords` (divided by wrong denominator)
**After:** F1-score (harmonic mean of precision & recall) ✅

### ❌ Problem 2: No Stemming
**Before:** "services" ≠ "service", "failing" ≠ "fail"
**After:** Both become "servic" and "fail" ✅

### ❌ Problem 3: Removed Question Words
**Before:** Filtered "why", "which", "how" (lost context)
**After:** Kept question words ✅

### ❌ Problem 4: Mixed Scores
**Before:** `avg([primary_scores + enrichment_scores])` - enrichments masked bad primaries
**After:** `0.8 * primary + 0.2 * enrichment` ✅

### ❌ Problem 5: Hardcoded Weights
**Before:** Fixed 40/60 split
**After:** Configurable parameters ✅

## Understanding Low Confidence

**Example:** "which services are failing right now?"
- Intent: SERVICE_HEALTH
- Confidence: 0.29 (LOW) ✅

**Why LOW?**
- Query uses: "**failing**"
- Examples have: "**healthy**", "**doing**", "**status**"
- No examples with negative framing!

**The Solution:** Add examples to `intent_categories.yaml`:
```yaml
SERVICE_HEALTH:
  examples:
    - "Is payment-api healthy?"
    - "Which services are failing?"      # ADD THIS
    - "What services are down?"          # ADD THIS
    - "Show me broken services"          # ADD THIS
```

**Result:** Confidence improves to MEDIUM/HIGH! 🎯

## Key Takeaways

✅ **Low confidence is GOOD feedback** - tells you where intent definitions need improvement
✅ **Algorithm is fast & free** - no extra LLM calls
✅ **Mathematically sound** - uses F1-score, not broken formulas
✅ **Transparent** - shows exactly why scores are high/low
✅ **Production ready** - all critical issues fixed

## File Structure

```
intent_confidence.py           # Core confidence mapper
test_confidence_offline.py     # Offline tests (no AWS)
test_confidence.py             # Live tests (with AWS)
test_specific_query.py         # Debug specific queries
CONFIDENCE_README.md           # This file
```

## API Reference

### `IntentConfidenceMapper`

```python
# Initialize
mapper = IntentConfidenceMapper(
    intent_categories: Dict,
    enrichment_rules: Dict,
    keyword_weight: float = 0.4,
    similarity_weight: float = 0.6
)

# Score primary intent
score, details = mapper.calculate_primary_intent_confidence(
    query: str,
    intent: str
) -> Tuple[float, Dict]

# Validate enrichment
score, details = mapper.calculate_enrichment_confidence(
    primary_intent: str,
    enrichment_intent: str
) -> Tuple[float, Dict]

# Validate full result
validation = mapper.validate_classification_result(
    query: str,
    primary_intents: List[str],
    enriched_intents: List[str]
) -> Dict
```

### `calculate_token_consumption()`

```python
from intent_confidence import calculate_token_consumption

tokens = calculate_token_consumption(
    system_prompt: str,
    user_query: str,
    response: str,
    model_id: str = "claude-3-5-sonnet"
) -> Dict
```

## Support

For issues or questions, check:
1. Run offline tests: `python3 test_confidence_offline.py`
2. Test specific query: `python3 test_specific_query.py`
3. Review intent examples in `intent_categories.yaml`
4. Check enrichment rules in `enrichment_rules.yaml`
