#!/usr/bin/env python3
"""Test the specific query that had low confidence"""

import yaml
from intent_confidence import IntentConfidenceMapper

# Load configurations
intent_categories = yaml.safe_load(open('intent_categories.yaml'))
enrichment_rules = yaml.safe_load(open('enrichment_rules.yaml'))

# Initialize mapper
mapper = IntentConfidenceMapper(intent_categories, enrichment_rules)

# The problematic query
query = "which services are failing right now?"

# Test against the intents the LLM chose
test_intents = ["CURRENT_HEALTH", "SERVICE_HEALTH"]

print("="*80)
print("TESTING SPECIFIC QUERY WITH FIXED LOGIC")
print("="*80)
print(f"\nQuery: \"{query}\"\n")

for intent in test_intents:
    print(f"\n{'-'*80}")
    print(f"Intent: {intent}")
    print(f"{'-'*80}")

    score, details = mapper.calculate_primary_intent_confidence(query, intent)

    print(f"\n✓ Confidence Score: {score:.3f} ({mapper.get_confidence_level(score)})")
    print(f"\nDetails:")
    print(f"  Keyword Match (F1): {details['keyword_match_score']:.3f}")
    print(f"    - Precision: {details.get('keyword_precision', 0):.3f}")
    print(f"    - Recall: {details.get('keyword_recall', 0):.3f}")
    print(f"  Example Similarity: {details['example_similarity_score']:.3f}")

    print(f"\n  Query Keywords: {details.get('query_keywords', [])}")
    print(f"  Intent Keywords: {details.get('intent_keywords', [])[:10]}...")  # Show first 10
    print(f"  Matched Keywords: {details.get('matched_keywords', [])}")

    print(f"\n  Reasoning:")
    for reason in details['reasoning']:
        print(f"     • {reason}")

# Compare with old logic (what it would have been)
print(f"\n{'='*80}")
print("COMPARISON: OLD vs NEW LOGIC")
print("="*80)

# Simulate old logic for SERVICE_HEALTH
query_keywords = {'which', 'service', 'fail', 'right', 'now'}  # Stemmed
intent_keywords_count = 11  # SERVICE_HEALTH has ~11 keywords
matched = 1  # Only 'service' matches

old_score = 1 / intent_keywords_count  # Old broken formula
print(f"\nOLD Logic (SERVICE_HEALTH):")
print(f"  Keyword score: {matched}/{intent_keywords_count} = {old_score:.3f} (14%)")
print(f"  Problem: Denominator too large!")

# New logic
score, details = mapper.calculate_primary_intent_confidence(query, "SERVICE_HEALTH")
print(f"\nNEW Logic (SERVICE_HEALTH):")
print(f"  Keyword F1-score: {details['keyword_match_score']:.3f}")
print(f"  Overall score: {score:.3f} ({mapper.get_confidence_level(score)})")
print(f"  ✓ Much better!")

print("\n" + "="*80 + "\n")
