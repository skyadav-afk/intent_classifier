#!/usr/bin/env python3
"""
Test script for Intent Confidence Mapping System
Tests confidence scoring for primary intents and enrichments
"""

from intent_classifier import IntentClassifier


def test_confidence_system():
    """Test the confidence mapping system"""
    print("="*80)
    print("TESTING INTENT CONFIDENCE MAPPING SYSTEM")
    print("="*80)
    print("\nInitializing classifier with confidence mapping...")

    try:
        classifier = IntentClassifier()
        print("✅ Classifier initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return

    # Test queries with different confidence levels
    test_cases = [
        {
            "query": "Why is payment-api failing?",
            "expected_intent": "ROOT_CAUSE_SINGLE",
            "description": "Clear query matching example"
        },
        {
            "query": "Is payment-api healthy?",
            "expected_intent": "SERVICE_HEALTH",
            "description": "Exact match to example"
        },
        {
            "query": "What alerts are active and what incidents are open?",
            "expected_intents": ["ALERT_STATUS", "INCIDENT_STATUS"],
            "description": "Multi-intent query"
        },
        {
            "query": "Show me the current status",
            "expected_intent": "CURRENT_HEALTH",
            "description": "Vague query - lower confidence expected"
        },
        {
            "query": "How is everything doing right now?",
            "expected_intent": "CURRENT_HEALTH",
            "description": "Ambiguous query"
        }
    ]

    print(f"Running {len(test_cases)} test cases...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Query: \"{test_case['query']}\"")

        try:
            # Classify with confidence
            result = classifier.classify(test_case['query'], include_confidence=True)

            # Print full result
            classifier.print_result(result, show_confidence=True, show_tokens=True)

            # Additional confidence analysis
            if 'confidence_validation' in result:
                print("\n📊 Detailed Confidence Analysis:")

                # Primary intents analysis
                print("\n   Primary Intents Breakdown:")
                for intent, data in result['confidence_validation']['primary_intents'].items():
                    print(f"\n   {intent}:")
                    details = data['details']
                    print(f"      Valid: {details['is_valid']}")
                    print(f"      Keyword Match: {details['keyword_match_score']:.2f}")
                    print(f"      Example Similarity: {details['example_similarity_score']:.2f}")
                    print(f"      Final Score: {details['final_score']:.2f}")
                    if details['matched_keywords']:
                        print(f"      Matched Keywords: {', '.join(details['matched_keywords'])}")
                    print(f"      Reasoning:")
                    for reason in details['reasoning']:
                        print(f"         • {reason}")

                # Enrichments analysis
                if result['confidence_validation']['enrichments']:
                    print("\n   Enrichment Validation:")
                    for enrich, data in result['confidence_validation']['enrichments'].items():
                        print(f"\n   {enrich}:")
                        if data['sources']:
                            for source in data['sources']:
                                print(f"      From: {source['primary']} (confidence: {source['score']:.2f})")
                                for reason in source['details']['reasoning']:
                                    print(f"         • {reason}")
                        else:
                            print(f"      ⚠️  No valid enrichment source found!")

        except Exception as e:
            print(f"\n❌ Error processing test case: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTotal test cases: {len(test_cases)}")
    print("\n✅ Confidence system testing complete!")
    print("\nKey Features Demonstrated:")
    print("   • Primary intent confidence scoring (keyword + example similarity)")
    print("   • Enrichment validation (checks enrichment rules)")
    print("   • Overall confidence calculation")
    print("   • Token consumption tracking")
    print("   • Detailed reasoning for each confidence score")
    print("\n" + "="*80 + "\n")


def test_token_calculation():
    """Test token consumption calculation"""
    from intent_confidence import calculate_token_consumption

    print("\n" + "="*80)
    print("TESTING TOKEN CONSUMPTION CALCULATOR")
    print("="*80)

    # Test with sample data
    system_prompt = "You are a helpful assistant." * 50  # ~150 tokens
    user_query = "What is the weather like today?"  # ~7 tokens
    response = "The weather is sunny and warm." * 3  # ~21 tokens

    tokens = calculate_token_consumption(
        system_prompt=system_prompt,
        user_query=user_query,
        response=response,
        model_id="claude-3-5-sonnet"
    )

    print("\nSample Token Calculation:")
    print(f"   System Prompt: {len(system_prompt)} chars → ~{tokens['breakdown']['system_prompt_tokens']} tokens")
    print(f"   User Query: {len(user_query)} chars → ~{tokens['breakdown']['user_query_tokens']} tokens")
    print(f"   Response: {len(response)} chars → ~{tokens['breakdown']['response_tokens']} tokens")
    print(f"\nTotal Input Tokens: {tokens['input_tokens']}")
    print(f"Total Output Tokens: {tokens['output_tokens']}")
    print(f"Total Tokens: {tokens['total_tokens']}")
    print(f"\nEstimated Cost:")
    print(f"   Input: ${tokens['cost_usd']['input']:.6f}")
    print(f"   Output: ${tokens['cost_usd']['output']:.6f}")
    print(f"   Total: ${tokens['cost_usd']['total']:.6f}")
    print(f"\nNote: {tokens['note']}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test confidence system
    test_confidence_system()

    # Test token calculator
    test_token_calculation()
