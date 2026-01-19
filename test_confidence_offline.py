#!/usr/bin/env python3
"""
Offline test for confidence mapping (no LLM calls)
Tests the confidence scoring logic independently
"""

import yaml
from intent_confidence import IntentConfidenceMapper, calculate_token_consumption


def load_yaml(filename: str) -> dict:
    """Load YAML file"""
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def test_confidence_mapper_offline():
    """Test confidence mapper without LLM calls"""
    print("="*80)
    print("OFFLINE CONFIDENCE MAPPER TEST")
    print("="*80)

    # Load configurations
    print("\nLoading configurations...")
    intent_categories = load_yaml('intent_categories.yaml')
    enrichment_rules = load_yaml('enrichment_rules.yaml')
    print("✅ Configurations loaded\n")

    # Initialize confidence mapper
    print("Initializing confidence mapper...")
    mapper = IntentConfidenceMapper(intent_categories, enrichment_rules)
    print(f"✅ Confidence mapper initialized")
    print(f"   Valid intents: {len(mapper.all_valid_intents)}")
    print(f"   Intent keywords mapped: {len(mapper.intent_keywords)}")
    print(f"   Intent examples mapped: {len(mapper.intent_examples)}\n")

    # Test cases
    test_cases = [
        {
            "query": "Why is payment-api failing?",
            "primary_intent": "ROOT_CAUSE_SINGLE",
            "description": "Exact match to example"
        },
        {
            "query": "Is payment-api healthy?",
            "primary_intent": "SERVICE_HEALTH",
            "description": "Very similar to example"
        },
        {
            "query": "Show me current status",
            "primary_intent": "CURRENT_HEALTH",
            "description": "Partial keyword match"
        },
        {
            "query": "Something is broken",
            "primary_intent": "ROOT_CAUSE_SINGLE",
            "description": "Vague query - low confidence expected"
        },
        {
            "query": "Invalid intent test",
            "primary_intent": "INVALID_INTENT",
            "description": "Invalid intent name"
        }
    ]

    print("="*80)
    print("PRIMARY INTENT CONFIDENCE TESTING")
    print("="*80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case['description']}")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Intent: {test_case['primary_intent']}")

        score, details = mapper.calculate_primary_intent_confidence(
            query=test_case['query'],
            intent=test_case['primary_intent']
        )

        print(f"\n✓ Confidence Score: {score:.3f} ({mapper.get_confidence_level(score)})")
        print(f"  Valid Intent: {details['is_valid']}")
        print(f"  Keyword Match Score: {details['keyword_match_score']:.3f}")
        print(f"  Example Similarity Score: {details['example_similarity_score']:.3f}")

        if details['matched_keywords']:
            print(f"  Matched Keywords: {', '.join(details['matched_keywords'])}")

        print(f"  Reasoning:")
        for reason in details['reasoning']:
            print(f"     • {reason}")
        print("-" * 80)

    # Test enrichment validation
    print("\n" + "="*80)
    print("ENRICHMENT VALIDATION TESTING")
    print("="*80)

    enrichment_tests = [
        {
            "primary": "ROOT_CAUSE_SINGLE",
            "enrichment": "UNDERCURRENTS_TREND",
            "description": "Valid direct enrichment"
        },
        {
            "primary": "ROOT_CAUSE_SINGLE",
            "enrichment": "MITIGATION_STEPS",
            "description": "Valid direct enrichment"
        },
        {
            "primary": "UNDERCURRENTS_TREND",
            "enrichment": "RISK_PREDICTION",
            "description": "Valid direct enrichment"
        },
        {
            "primary": "ROOT_CAUSE_SINGLE",
            "enrichment": "ALERT_STATUS",
            "description": "Invalid enrichment (not linked)"
        },
        {
            "primary": "ROOT_CAUSE_SINGLE",
            "enrichment": "INVALID_ENRICHMENT",
            "description": "Invalid enrichment (doesn't exist)"
        }
    ]

    for i, test in enumerate(enrichment_tests, 1):
        print(f"\n[Test {i}] {test['description']}")
        print(f"Primary: {test['primary']} → Enrichment: {test['enrichment']}")

        score, details = mapper.calculate_enrichment_confidence(
            primary_intent=test['primary'],
            enrichment_intent=test['enrichment']
        )

        print(f"\n✓ Confidence Score: {score:.3f} ({mapper.get_confidence_level(score)})")
        print(f"  Valid Intent: {details['is_valid_intent']}")
        print(f"  Valid Enrichment: {details['is_valid_enrichment']}")
        print(f"  Enrichment Depth: {details['enrichment_depth']}")
        print(f"  Reasoning:")
        for reason in details['reasoning']:
            print(f"     • {reason}")
        print("-" * 80)

    # Test complete validation
    print("\n" + "="*80)
    print("COMPLETE CLASSIFICATION VALIDATION")
    print("="*80)

    query = "Why is payment-api failing?"
    primary_intents = ["ROOT_CAUSE_SINGLE"]
    enriched_intents = ["ROOT_CAUSE_SINGLE", "UNDERCURRENTS_TREND", "MITIGATION_STEPS"]

    print(f"\nQuery: \"{query}\"")
    print(f"Primary Intents: {primary_intents}")
    print(f"Enriched Intents: {enriched_intents}")

    validation = mapper.validate_classification_result(
        query=query,
        primary_intents=primary_intents,
        enriched_intents=enriched_intents
    )

    print(f"\n✓ Overall Confidence: {validation['overall_confidence']:.3f}")
    print(f"\nPrimary Intents Validation:")
    for intent, data in validation['primary_intents'].items():
        print(f"   {intent}: {data['confidence_score']:.3f} ({mapper.get_confidence_level(data['confidence_score'])})")

    print(f"\nEnrichments Validation:")
    for enrich, data in validation['enrichments'].items():
        print(f"   {enrich}: {data['max_confidence']:.3f} ({mapper.get_confidence_level(data['max_confidence'])})")

    print(f"\nRecommendations:")
    for rec in validation['recommendations']:
        print(f"   • {rec}")

    print("\n" + "="*80 + "\n")


def test_token_calculator():
    """Test token consumption calculator"""
    print("="*80)
    print("TOKEN CONSUMPTION CALCULATOR TEST")
    print("="*80)

    # Test scenarios
    scenarios = [
        {
            "name": "Small query",
            "system": "You are an intent classifier." * 10,
            "query": "Is payment-api healthy?",
            "response": '["SERVICE_HEALTH"]'
        },
        {
            "name": "Large system prompt",
            "system": "You are an intent classifier. " * 200,
            "query": "Why is payment-api failing?",
            "response": '["ROOT_CAUSE_SINGLE"]'
        },
        {
            "name": "Complex query",
            "system": "You are an intent classifier." * 10,
            "query": "What alerts are active and what incidents are open? Also show me which SLOs are breached.",
            "response": '["ALERT_STATUS", "INCIDENT_STATUS", "SLO_STATUS"]'
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 80)

        tokens = calculate_token_consumption(
            system_prompt=scenario['system'],
            user_query=scenario['query'],
            response=scenario['response'],
            model_id="claude-3-5-sonnet"
        )

        print(f"System Prompt: {len(scenario['system'])} chars → {tokens['breakdown']['system_prompt_tokens']} tokens")
        print(f"User Query: {len(scenario['query'])} chars → {tokens['breakdown']['user_query_tokens']} tokens")
        print(f"Response: {len(scenario['response'])} chars → {tokens['breakdown']['response_tokens']} tokens")
        print(f"\nTotal Input:  {tokens['input_tokens']:,} tokens (${tokens['cost_usd']['input']:.6f})")
        print(f"Total Output: {tokens['output_tokens']:,} tokens (${tokens['cost_usd']['output']:.6f})")
        print(f"Total Cost:   ${tokens['cost_usd']['total']:.6f}")

    print("\n" + "="*80)
    print("✅ All offline tests completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test confidence mapper
    test_confidence_mapper_offline()

    # Test token calculator
    test_token_calculator()
