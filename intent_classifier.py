#!/usr/bin/env python3
"""
Intent Classifier for Conversational SLO Manager
Uses AWS Bedrock to classify user queries into intents and determine data sources
"""

import os
import json
import yaml
from typing import Dict, List, Set, Any
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from intent_confidence import IntentConfidenceMapper, calculate_token_consumption


class IntentClassifier:
    """Main intent classifier class"""

    def __init__(self):
        """Initialize the intent classifier"""
        # Load environment variables
        load_dotenv()

        # Load YAML configurations
        self.intent_categories = self._load_yaml('intent_categories.yaml')
        self.enrichment_rules = self._load_yaml('enrichment_rules.yaml')
        self.data_sources_config = self._load_yaml('data_sources.yaml')

        # Build intent to data sources mapping
        self.intent_to_data_sources = self._build_intent_data_source_map()

        # Initialize AWS Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        # Model configuration
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '10'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.0'))

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

        # Initialize confidence mapper
        self.confidence_mapper = IntentConfidenceMapper(
            self.intent_categories,
            self.enrichment_rules
        )

        # Token tracking
        self.last_token_consumption = None

    def _load_yaml(self, filename: str) -> Dict:
        """Load YAML file"""
        try:
            with open(filename, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing {filename}: {e}")
            return {}

    def _build_intent_data_source_map(self) -> Dict[str, List[str]]:
        """Build mapping from intent to data sources"""
        intent_map = {}

        for category, category_data in self.intent_categories.items():
            if 'intents' in category_data:
                for intent_name, intent_data in category_data['intents'].items():
                    if 'data_sources' in intent_data:
                        intent_map[intent_name] = intent_data['data_sources']

        return intent_map

    def _build_system_prompt(self) -> str:
        """Build the system prompt for intent classification"""
        prompt = """You are an intent classifier for an enterprise observability and SRE platform.

Classify the user's question into ONE OR MORE of the following intents. Return multiple intents if the question requires information from different areas.

"""

        # Add all categories and their intents
        for category, category_data in self.intent_categories.items():
            if 'intents' in category_data:
                prompt += f"{category}:\n"
                for intent_name, intent_data in category_data['intents'].items():
                    prompt += f"- {intent_name}: {intent_data.get('description', '')}\n"
                    if 'examples' in intent_data and intent_data['examples']:
                        example = intent_data['examples'][0]
                        prompt += f"  Example: \"{example}\"\n"
                prompt += "\n"

        prompt += """Return your response as a JSON array of intent names ONLY.
Format: ["INTENT_NAME1", "INTENT_NAME2"]

If only one intent is detected, return: ["INTENT_NAME"]

Examples:
- "Why is payment-api failing?" → ["ROOT_CAUSE_SINGLE"]
- "What alerts are active and what incidents are open?" → ["ALERT_STATUS", "INCIDENT_STATUS"]
- "Are we meeting our SLOs and what's the trend?" → ["SLO_STATUS", "SLO_BURN_TREND"]
"""

        return prompt

    def _call_bedrock(self, user_query: str) -> tuple[List[str], str]:
        """
        Call AWS Bedrock to classify intent

        Returns:
            Tuple of (intents_list, llm_response_text)
        """
        try:
            # Prepare the request body for Claude 3
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": self.system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            }

            # Invoke the model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            assistant_message = response_body['content'][0]['text']

            # Calculate token consumption
            self.last_token_consumption = calculate_token_consumption(
                system_prompt=self.system_prompt,
                user_query=user_query,
                response=assistant_message,
                model_id=self.model_id
            )

            # Extract JSON array from response
            # Handle cases where LLM might add extra text
            assistant_message_clean = assistant_message.strip()

            # Find JSON array in the response
            start_idx = assistant_message_clean.find('[')
            end_idx = assistant_message_clean.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = assistant_message_clean[start_idx:end_idx]
                intents = json.loads(json_str)
                return (intents if isinstance(intents, list) else [intents], assistant_message)
            else:
                # Fallback: try to parse the entire response
                return (json.loads(assistant_message_clean), assistant_message)

        except ClientError as e:
            print(f"AWS Bedrock Error: {e}")
            return ([], "")
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
            print(f"LLM Response: {assistant_message}")
            return ([], assistant_message if 'assistant_message' in locals() else "")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ([], "")

    def _get_enrichment_intents(self, primary_intents: List[str]) -> Set[str]:
        """Get all enrichment intents for the primary intents"""
        enriched_intents = set(primary_intents)

        # Process each primary intent
        for intent in primary_intents:
            if intent in self.enrichment_rules:
                # Add all enrichment intents
                for enrichment in self.enrichment_rules[intent]:
                    enriched_intents.add(enrichment)

        return enriched_intents

    def _get_data_sources(self, intents: Set[str]) -> List[str]:
        """Get all required data sources for the intents"""
        data_sources = set()

        for intent in intents:
            if intent in self.intent_to_data_sources:
                data_sources.update(self.intent_to_data_sources[intent])

        return sorted(list(data_sources))

    def classify(self, user_query: str, include_confidence: bool = True) -> Dict[str, Any]:
        """
        Classify user query and return intent, enrichments, and data sources

        Args:
            user_query: The user's question
            include_confidence: Whether to include confidence scoring (default: True)

        Returns:
            Dictionary containing:
            - primary_intents: List of detected intents
            - enriched_intents: All intents including enrichments
            - data_sources: Required data sources
            - enrichment_details: Details of which enrichments came from which primary intent
            - confidence_validation: Confidence scores for intents (if include_confidence=True)
            - token_consumption: Token usage statistics
        """
        # Get primary intents from LLM
        primary_intents, llm_response = self._call_bedrock(user_query)

        if not primary_intents:
            return {
                "error": "Failed to classify intent",
                "primary_intents": [],
                "enriched_intents": [],
                "data_sources": [],
                "enrichment_details": {},
                "token_consumption": self.last_token_consumption
            }

        # Get enriched intents
        enriched_intents = self._get_enrichment_intents(primary_intents)

        # Build enrichment details
        enrichment_details = {}
        for intent in primary_intents:
            if intent in self.enrichment_rules:
                enrichment_details[intent] = self.enrichment_rules[intent]

        # Get required data sources
        data_sources = self._get_data_sources(enriched_intents)

        result = {
            "query": user_query,
            "primary_intents": primary_intents,
            "enriched_intents": sorted(list(enriched_intents)),
            "data_sources": data_sources,
            "enrichment_details": enrichment_details,
            "token_consumption": self.last_token_consumption
        }

        # Add confidence validation if requested
        if include_confidence:
            confidence_validation = self.confidence_mapper.validate_classification_result(
                query=user_query,
                primary_intents=primary_intents,
                enriched_intents=sorted(list(enriched_intents))
            )
            result["confidence_validation"] = confidence_validation

        return result

    def print_result(self, result: Dict[str, Any], show_confidence: bool = True, show_tokens: bool = True):
        """
        Pretty print the classification result

        Args:
            result: Classification result dictionary
            show_confidence: Whether to show confidence scores (default: True)
            show_tokens: Whether to show token consumption (default: True)
        """
        if "error" in result:
            print(f"\n❌ Error: {result['error']}\n")
            return

        print("\n" + "="*80)
        print("INTENT CLASSIFICATION RESULT")
        print("="*80)

        print(f"\n📝 Query: {result['query']}")

        # Print primary intents with confidence
        print(f"\n🎯 Primary Intent(s):")
        if show_confidence and 'confidence_validation' in result:
            for intent in result['primary_intents']:
                conf_data = result['confidence_validation']['primary_intents'].get(intent, {})
                score = conf_data.get('confidence_score', 0.0)
                level = self.confidence_mapper.get_confidence_level(score)
                print(f"   • {intent} [Confidence: {score:.2f} - {level}]")
        else:
            for intent in result['primary_intents']:
                print(f"   • {intent}")

        # Print enrichment details
        if result['enrichment_details']:
            print(f"\n🔄 Enrichment Applied:")
            for primary, enrichments in result['enrichment_details'].items():
                print(f"   {primary} →")
                for enrich in enrichments:
                    if show_confidence and 'confidence_validation' in result:
                        enrich_data = result['confidence_validation']['enrichments'].get(enrich, {})
                        max_conf = enrich_data.get('max_confidence', 0.0)
                        level = self.confidence_mapper.get_confidence_level(max_conf)
                        print(f"      • {enrich} [Confidence: {max_conf:.2f} - {level}]")
                    else:
                        print(f"      • {enrich}")

        # Print all intents
        print(f"\n📊 All Intents (including enrichments):")
        for intent in result['enriched_intents']:
            marker = "🎯" if intent in result['primary_intents'] else "  "
            print(f"   {marker} {intent}")

        # Print data sources
        print(f"\n💾 Data Sources Required:")
        for ds in result['data_sources']:
            # Get description from config
            ds_info = self.data_sources_config.get('data_sources', {}).get(ds, {})
            description = ds_info.get('description', 'No description')
            print(f"   • {ds}: {description}")

        # Print confidence summary
        if show_confidence and 'confidence_validation' in result:
            conf = result['confidence_validation']
            print(f"\n🔍 Confidence Summary:")
            print(f"   Overall Confidence: {conf['overall_confidence']:.2f}")
            if conf.get('recommendations'):
                print(f"   Recommendations:")
                for rec in conf['recommendations'][:3]:  # Show top 3
                    print(f"      • {rec}")

        # Print token consumption
        if show_tokens and result.get('token_consumption'):
            tokens = result['token_consumption']
            print(f"\n💰 Token Consumption:")
            print(f"   Input tokens:  {tokens['input_tokens']:,} (${tokens['cost_usd']['input']:.6f})")
            print(f"   Output tokens: {tokens['output_tokens']:,} (${tokens['cost_usd']['output']:.6f})")
            print(f"   Total tokens:  {tokens['total_tokens']:,} (${tokens['cost_usd']['total']:.6f})")

        print("\n" + "="*80 + "\n")


def main():
    """Main function for interactive testing"""
    print("="*80)
    print("CONVERSATIONAL SLO MANAGER - INTENT CLASSIFIER")
    print("="*80)
    print("\nInitializing classifier...")

    try:
        classifier = IntentClassifier()
        print("✅ Classifier initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return

    print("Enter your queries (type 'quit' or 'exit' to stop):\n")

    while True:
        try:
            user_input = input("Query: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            # Classify the query
            result = classifier.classify(user_input)
            classifier.print_result(result)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")


if __name__ == "__main__":
    main()
