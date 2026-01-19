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

    def _call_bedrock(self, user_query: str) -> List[str]:
        """Call AWS Bedrock to classify intent"""
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

            # Extract JSON array from response
            # Handle cases where LLM might add extra text
            assistant_message = assistant_message.strip()

            # Find JSON array in the response
            start_idx = assistant_message.find('[')
            end_idx = assistant_message.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = assistant_message[start_idx:end_idx]
                intents = json.loads(json_str)
                return intents if isinstance(intents, list) else [intents]
            else:
                # Fallback: try to parse the entire response
                return json.loads(assistant_message)

        except ClientError as e:
            print(f"AWS Bedrock Error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
            print(f"LLM Response: {assistant_message}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

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

    def classify(self, user_query: str) -> Dict[str, Any]:
        """
        Classify user query and return intent, enrichments, and data sources

        Args:
            user_query: The user's question

        Returns:
            Dictionary containing:
            - primary_intents: List of detected intents
            - enriched_intents: All intents including enrichments
            - data_sources: Required data sources
            - enrichment_details: Details of which enrichments came from which primary intent
        """
        # Get primary intents from LLM
        primary_intents = self._call_bedrock(user_query)

        if not primary_intents:
            return {
                "error": "Failed to classify intent",
                "primary_intents": [],
                "enriched_intents": [],
                "data_sources": [],
                "enrichment_details": {}
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

        return {
            "query": user_query,
            "primary_intents": primary_intents,
            "enriched_intents": sorted(list(enriched_intents)),
            "data_sources": data_sources,
            "enrichment_details": enrichment_details
        }

    def print_result(self, result: Dict[str, Any]):
        """Pretty print the classification result"""
        if "error" in result:
            print(f"\n❌ Error: {result['error']}\n")
            return

        print("\n" + "="*80)
        print("INTENT CLASSIFICATION RESULT")
        print("="*80)

        print(f"\n📝 Query: {result['query']}")

        print(f"\n🎯 Primary Intent(s): {', '.join(result['primary_intents'])}")

        if result['enrichment_details']:
            print(f"\n🔄 Enrichment Applied:")
            for primary, enrichments in result['enrichment_details'].items():
                print(f"   {primary} → {', '.join(enrichments)}")

        print(f"\n📊 All Intents (including enrichments):")
        for intent in result['enriched_intents']:
            marker = "🎯" if intent in result['primary_intents'] else "  "
            print(f"   {marker} {intent}")

        print(f"\n💾 Data Sources Required:")
        for ds in result['data_sources']:
            # Get description from config
            ds_info = self.data_sources_config.get('data_sources', {}).get(ds, {})
            description = ds_info.get('description', 'No description')
            print(f"   • {ds}: {description}")

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
