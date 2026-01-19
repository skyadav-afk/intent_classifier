# Understanding intent_classifier.py - Deep Dive

This document provides a comprehensive explanation of the `intent_classifier.py` implementation, breaking down every component, method, and design decision.

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Code Structure](#code-structure)
3. [Class: IntentClassifier](#class-intentclassifier)
4. [Initialization Flow](#initialization-flow)
5. [Classification Pipeline](#classification-pipeline)
6. [AWS Bedrock Integration](#aws-bedrock-integration)
7. [Enrichment Logic](#enrichment-logic)
8. [Data Source Mapping](#data-source-mapping)
9. [Main Interactive Loop](#main-interactive-loop)
10. [Error Handling](#error-handling)

---

## High-Level Overview

The `intent_classifier.py` file implements a conversational AI system that:
1. Takes natural language queries from users
2. Classifies them into specific operational intents (e.g., "root cause analysis", "SLO status")
3. Enriches the classification with related intents
4. Determines which data sources are needed to answer the query

**Key Technologies:**
- **AWS Bedrock**: Cloud-based LLM service (using Claude 3.5 Sonnet)
- **YAML**: Configuration-driven design
- **Python 3**: Core implementation language

---

## Code Structure

```
intent_classifier.py
├── Imports & Dependencies
├── IntentClassifier Class
│   ├── __init__()                          # Initialization
│   ├── _load_yaml()                        # YAML loader
│   ├── _build_intent_data_source_map()     # Build intent→data source mapping
│   ├── _build_system_prompt()              # Generate LLM prompt
│   ├── _call_bedrock()                     # AWS Bedrock API call
│   ├── _get_enrichment_intents()           # Apply enrichment rules
│   ├── _get_data_sources()                 # Collect required data sources
│   ├── classify()                          # Main classification method
│   └── print_result()                      # Pretty-print results
└── main()                                  # Interactive CLI entry point
```

---

## Class: IntentClassifier

### Purpose
The `IntentClassifier` class is the core of the system. It encapsulates all logic for:
- Loading configuration files
- Building internal mappings
- Communicating with AWS Bedrock
- Processing and enriching classifications

### Class Attributes

```python
self.intent_categories       # Dict: Loaded from intent_categories.yaml
self.enrichment_rules        # Dict: Loaded from enrichment_rules.yaml
self.data_sources_config     # Dict: Loaded from data_sources.yaml
self.intent_to_data_sources  # Dict: Maps intent → [data_sources]
self.bedrock_runtime         # boto3 client for AWS Bedrock
self.model_id                # AWS Bedrock model identifier
self.max_tokens              # LLM response length limit
self.temperature             # LLM randomness (0.0 = deterministic)
self.system_prompt           # Pre-built prompt for LLM
```

---

## Initialization Flow

### Step-by-Step Breakdown

```python
def __init__(self):
    """Initialize the intent classifier"""
```

#### Step 1: Load Environment Variables
```python
load_dotenv()
```
- Reads `.env` file in the current directory
- Loads AWS credentials and configuration into environment variables
- **Why?** Keeps sensitive credentials out of source code

#### Step 2: Load YAML Configurations
```python
self.intent_categories = self._load_yaml('intent_categories.yaml')
self.enrichment_rules = self._load_yaml('enrichment_rules.yaml')
self.data_sources_config = self._load_yaml('data_sources.yaml')
```

**What happens in `_load_yaml()`:**
```python
def _load_yaml(self, filename: str) -> Dict:
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing {filename}: {e}")
        return {}
```

- Opens YAML file
- Parses it into a Python dictionary
- Returns empty dict on error (graceful failure)
- **Why safe_load?** Prevents arbitrary code execution from YAML

#### Step 3: Build Intent-to-Data-Source Mapping
```python
self.intent_to_data_sources = self._build_intent_data_source_map()
```

**What `_build_intent_data_source_map()` does:**
```python
def _build_intent_data_source_map(self) -> Dict[str, List[str]]:
    intent_map = {}

    for category, category_data in self.intent_categories.items():
        if 'intents' in category_data:
            for intent_name, intent_data in category_data['intents'].items():
                if 'data_sources' in intent_data:
                    intent_map[intent_name] = intent_data['data_sources']

    return intent_map
```

**Example transformation:**
```yaml
# Input (intent_categories.yaml)
STATE:
  intents:
    CURRENT_HEALTH:
      data_sources:
        - java_stats_api
        - postgres
```

```python
# Output (Python dict)
{
    'CURRENT_HEALTH': ['java_stats_api', 'postgres']
}
```

**Why?** Fast O(1) lookup when we need to find data sources for an intent.

#### Step 4: Initialize AWS Bedrock Client
```python
self.bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
```

- Creates boto3 client for AWS Bedrock Runtime
- Uses credentials from environment variables
- Default region: `us-east-1` if not specified

#### Step 5: Configure Model Parameters
```python
self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
self.max_tokens = int(os.getenv('MAX_TOKENS', '500'))
self.temperature = float(os.getenv('TEMPERATURE', '0.0'))
```

- **model_id**: Which Claude model to use
- **max_tokens**: Maximum response length (500 is sufficient for intent classification)
- **temperature**: 0.0 = deterministic, 1.0 = creative

**Why temperature=0.0?** We want consistent, predictable classifications.

#### Step 6: Build System Prompt
```python
self.system_prompt = self._build_system_prompt()
```

This generates the instruction prompt for the LLM. Details in [AWS Bedrock Integration](#aws-bedrock-integration).

---

## Classification Pipeline

### Main Entry Point: `classify()`

```python
def classify(self, user_query: str) -> Dict[str, Any]:
```

**Purpose:** Orchestrates the entire classification process.

### Step-by-Step Flow

```
User Query
    ↓
1. Call AWS Bedrock (_call_bedrock)
    ↓
Primary Intents: ["ROOT_CAUSE_SINGLE"]
    ↓
2. Apply Enrichment (_get_enrichment_intents)
    ↓
Enriched Intents: ["ROOT_CAUSE_SINGLE", "UNDERCURRENTS_TREND", "MITIGATION_STEPS"]
    ↓
3. Get Data Sources (_get_data_sources)
    ↓
Data Sources: ["java_stats_api", "opensearch", "clickhouse"]
    ↓
4. Build Response Object
    ↓
Return Complete Classification
```

### Code Walkthrough

```python
def classify(self, user_query: str) -> Dict[str, Any]:
    # Step 1: Get primary intents from LLM
    primary_intents = self._call_bedrock(user_query)

    # Error handling
    if not primary_intents:
        return {
            "error": "Failed to classify intent",
            "primary_intents": [],
            "enriched_intents": [],
            "data_sources": [],
            "enrichment_details": {}
        }

    # Step 2: Get enriched intents
    enriched_intents = self._get_enrichment_intents(primary_intents)

    # Step 3: Build enrichment details
    enrichment_details = {}
    for intent in primary_intents:
        if intent in self.enrichment_rules:
            enrichment_details[intent] = self.enrichment_rules[intent]

    # Step 4: Get required data sources
    data_sources = self._get_data_sources(enriched_intents)

    # Step 5: Return complete result
    return {
        "query": user_query,
        "primary_intents": primary_intents,
        "enriched_intents": sorted(list(enriched_intents)),
        "data_sources": data_sources,
        "enrichment_details": enrichment_details
    }
```

**Return Format Example:**
```python
{
    "query": "Why is payment-api failing?",
    "primary_intents": ["ROOT_CAUSE_SINGLE"],
    "enriched_intents": ["MITIGATION_STEPS", "ROOT_CAUSE_SINGLE", "UNDERCURRENTS_TREND"],
    "data_sources": ["clickhouse", "java_stats_api", "opensearch"],
    "enrichment_details": {
        "ROOT_CAUSE_SINGLE": ["UNDERCURRENTS_TREND", "MITIGATION_STEPS"]
    }
}
```

---

## AWS Bedrock Integration

### Building the System Prompt

```python
def _build_system_prompt(self) -> str:
```

**Purpose:** Generates the instruction prompt that tells Claude how to classify intents.

#### Prompt Structure

```
1. Role Definition
   "You are an intent classifier for an enterprise observability and SRE platform."

2. Task Description
   "Classify the user's question into ONE OR MORE of the following intents..."

3. Intent Catalog
   STATE:
   - CURRENT_HEALTH: Overall application health status
     Example: "How is my application now?"
   - SERVICE_HEALTH: Individual service health check
     Example: "Is payment-api healthy?"
   [... all 30+ intents with descriptions and examples ...]

4. Output Format
   "Return your response as a JSON array of intent names ONLY.
    Format: ["INTENT_NAME1", "INTENT_NAME2"]"

5. Few-Shot Examples
   - "Why is payment-api failing?" → ["ROOT_CAUSE_SINGLE"]
   - "What alerts are active and what incidents are open?" → ["ALERT_STATUS", "INCIDENT_STATUS"]
```

**Why this structure?**
- **Role**: Sets context for the LLM
- **Intent catalog**: Provides all available options with examples
- **Output format**: Ensures structured, parseable responses
- **Few-shot examples**: Demonstrates expected behavior

### Calling AWS Bedrock

```python
def _call_bedrock(self, user_query: str) -> List[str]:
```

#### Request Body Structure

```python
request_body = {
    "anthropic_version": "bedrock-2023-05-31",  # API version
    "max_tokens": self.max_tokens,               # Response length limit
    "temperature": self.temperature,             # Randomness (0.0)
    "system": self.system_prompt,                # Instructions
    "messages": [
        {
            "role": "user",
            "content": user_query                # Actual user question
        }
    ]
}
```

#### Making the API Call

```python
response = self.bedrock_runtime.invoke_model(
    modelId=self.model_id,
    body=json.dumps(request_body)
)
```

- **invoke_model**: Synchronous API call to Bedrock
- **modelId**: Specifies which Claude model to use
- **body**: JSON-encoded request

#### Parsing the Response

```python
response_body = json.loads(response['body'].read())
assistant_message = response_body['content'][0]['text']
```

**Response format from Bedrock:**
```python
{
    'content': [
        {
            'type': 'text',
            'text': '["ROOT_CAUSE_SINGLE"]'
        }
    ],
    'usage': {...},
    'stop_reason': 'end_turn'
}
```

#### Extracting JSON Array

```python
# Handle cases where LLM might add extra text
assistant_message = assistant_message.strip()

# Find JSON array in the response
start_idx = assistant_message.find('[')
end_idx = assistant_message.rfind(']') + 1

if start_idx != -1 and end_idx > start_idx:
    json_str = assistant_message[start_idx:end_idx]
    intents = json.loads(json_str)
    return intents if isinstance(intents, list) else [intents]
```

**Why this approach?**
- LLM might return: `"Here are the intents: ["ROOT_CAUSE_SINGLE"]"`
- We extract just the JSON array: `["ROOT_CAUSE_SINGLE"]`
- Robust to variations in LLM output

#### Error Handling

```python
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
```

**Three levels of error handling:**
1. **AWS errors**: Network, authentication, rate limits
2. **JSON errors**: Malformed LLM responses
3. **Generic errors**: Unexpected issues

---

## Enrichment Logic

### Purpose
Automatically include related intents to provide comprehensive answers.

### Implementation

```python
def _get_enrichment_intents(self, primary_intents: List[str]) -> Set[str]:
    """Get all enrichment intents for the primary intents"""
    enriched_intents = set(primary_intents)  # Start with primary intents

    # Process each primary intent
    for intent in primary_intents:
        if intent in self.enrichment_rules:
            # Add all enrichment intents
            for enrichment in self.enrichment_rules[intent]:
                enriched_intents.add(enrichment)

    return enriched_intents
```

### Example Flow

**Input:**
```python
primary_intents = ["ROOT_CAUSE_SINGLE"]
```

**Enrichment Rules (from enrichment_rules.yaml):**
```yaml
ROOT_CAUSE_SINGLE:
  - UNDERCURRENTS_TREND
  - MITIGATION_STEPS
```

**Process:**
```python
enriched_intents = {"ROOT_CAUSE_SINGLE"}  # Start with primary

# Iterate primary intents
for intent in ["ROOT_CAUSE_SINGLE"]:
    # Found in enrichment_rules
    for enrichment in ["UNDERCURRENTS_TREND", "MITIGATION_STEPS"]:
        enriched_intents.add(enrichment)

# Result: {"ROOT_CAUSE_SINGLE", "UNDERCURRENTS_TREND", "MITIGATION_STEPS"}
```

**Why use a Set?**
- Automatically prevents duplicates
- If multiple primary intents suggest the same enrichment, it's only added once

### Multi-Intent Enrichment Example

**Input:**
```python
primary_intents = ["ALERT_DEBUG", "SLO_STATUS"]
```

**Enrichment Rules:**
```yaml
ALERT_DEBUG:
  - ROOT_CAUSE_SINGLE
  - BLAST_RADIUS
  - MITIGATION_STEPS

SLO_STATUS:
  - SLO_BURN_TREND
  - RISK_PREDICTION
```

**Output:**
```python
{
    "ALERT_DEBUG",
    "SLO_STATUS",
    "ROOT_CAUSE_SINGLE",      # from ALERT_DEBUG
    "BLAST_RADIUS",           # from ALERT_DEBUG
    "MITIGATION_STEPS",       # from ALERT_DEBUG
    "SLO_BURN_TREND",         # from SLO_STATUS
    "RISK_PREDICTION"         # from SLO_STATUS
}
```

---

## Data Source Mapping

### Purpose
Determine which databases/APIs are needed to answer the query.

### Implementation

```python
def _get_data_sources(self, intents: Set[str]) -> List[str]:
    """Get all required data sources for the intents"""
    data_sources = set()

    for intent in intents:
        if intent in self.intent_to_data_sources:
            data_sources.update(self.intent_to_data_sources[intent])

    return sorted(list(data_sources))
```

### Example Flow

**Input (enriched intents):**
```python
intents = {"ROOT_CAUSE_SINGLE", "UNDERCURRENTS_TREND", "MITIGATION_STEPS"}
```

**Intent-to-Data-Source Mapping:**
```python
{
    "ROOT_CAUSE_SINGLE": ["java_stats_api", "opensearch", "clickhouse"],
    "UNDERCURRENTS_TREND": ["java_stats_api", "clickhouse"],
    "MITIGATION_STEPS": ["clickhouse"]
}
```

**Process:**
```python
data_sources = set()

for intent in intents:
    # ROOT_CAUSE_SINGLE
    data_sources.update(["java_stats_api", "opensearch", "clickhouse"])
    # Now: {"java_stats_api", "opensearch", "clickhouse"}

    # UNDERCURRENTS_TREND
    data_sources.update(["java_stats_api", "clickhouse"])
    # No change (already present)

    # MITIGATION_STEPS
    data_sources.update(["clickhouse"])
    # No change (already present)

return sorted(list(data_sources))
# Result: ["clickhouse", "java_stats_api", "opensearch"]
```

**Why sorted?**
- Consistent output order
- Easier to test and debug

---

## Main Interactive Loop

### Entry Point

```python
def main():
    """Main function for interactive testing"""
```

### Initialization Phase

```python
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
```

**What can go wrong?**
- YAML files not found
- AWS credentials missing/invalid
- Network issues connecting to Bedrock

### Interactive Loop

```python
print("Enter your queries (type 'quit' or 'exit' to stop):\n")

while True:
    try:
        user_input = input("Query: ").strip()

        # Skip empty input
        if not user_input:
            continue

        # Exit conditions
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
```

**Key features:**
- Strips whitespace from input
- Ignores empty lines
- Multiple exit commands (quit/exit/q)
- Graceful Ctrl+C handling
- Error recovery (continues on exception)

---

## Error Handling

### Layered Approach

#### Layer 1: File Loading
```python
def _load_yaml(self, filename: str) -> Dict:
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing {filename}: {e}")
        return {}
```

**Strategy:** Return empty dict, allow system to continue (graceful degradation)

#### Layer 2: AWS API Calls
```python
def _call_bedrock(self, user_query: str) -> List[str]:
    try:
        # API call code
        ...
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
```

**Strategy:** Return empty list, let `classify()` handle the failure

#### Layer 3: Classification
```python
def classify(self, user_query: str) -> Dict[str, Any]:
    primary_intents = self._call_bedrock(user_query)

    if not primary_intents:
        return {
            "error": "Failed to classify intent",
            "primary_intents": [],
            "enriched_intents": [],
            "data_sources": [],
            "enrichment_details": {}
        }
```

**Strategy:** Return error object with consistent structure

#### Layer 4: User Interface
```python
def main():
    try:
        classifier = IntentClassifier()
        print("✅ Classifier initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return  # Exit if initialization fails

    while True:
        try:
            # Process query
            ...
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")
            # Continue loop (don't exit)
```

**Strategy:** Fail hard on initialization, recover from query errors

---

## Pretty Printing Results

### Implementation

```python
def print_result(self, result: Dict[str, Any]):
    """Pretty print the classification result"""

    # Handle errors
    if "error" in result:
        print(f"\n❌ Error: {result['error']}\n")
        return

    # Header
    print("\n" + "="*80)
    print("INTENT CLASSIFICATION RESULT")
    print("="*80)

    # Query
    print(f"\n📝 Query: {result['query']}")

    # Primary intents
    print(f"\n🎯 Primary Intent(s): {', '.join(result['primary_intents'])}")

    # Enrichment details
    if result['enrichment_details']:
        print(f"\n🔄 Enrichment Applied:")
        for primary, enrichments in result['enrichment_details'].items():
            print(f"   {primary} → {', '.join(enrichments)}")

    # All intents
    print(f"\n📊 All Intents (including enrichments):")
    for intent in result['enriched_intents']:
        marker = "🎯" if intent in result['primary_intents'] else "  "
        print(f"   {marker} {intent}")

    # Data sources
    print(f"\n💾 Data Sources Required:")
    for ds in result['data_sources']:
        ds_info = self.data_sources_config.get('data_sources', {}).get(ds, {})
        description = ds_info.get('description', 'No description')
        print(f"   • {ds}: {description}")

    print("\n" + "="*80 + "\n")
```

### Output Example

```
================================================================================
INTENT CLASSIFICATION RESULT
================================================================================

📝 Query: Why is payment-api failing?

🎯 Primary Intent(s): ROOT_CAUSE_SINGLE

🔄 Enrichment Applied:
   ROOT_CAUSE_SINGLE → UNDERCURRENTS_TREND, MITIGATION_STEPS

📊 All Intents (including enrichments):
   🎯 ROOT_CAUSE_SINGLE
      UNDERCURRENTS_TREND
      MITIGATION_STEPS

💾 Data Sources Required:
   • java_stats_api: Real-time metrics and statistics from Java services
   • opensearch: Logs, traces, and full-text search
   • clickhouse: Historical data, trends, patterns, and AI memory

================================================================================
```

**Design choices:**
- Emojis for visual categorization
- Clear hierarchy (headers, indentation)
- Distinguishes primary intents (🎯) from enrichments
- Shows data source descriptions from config
- Consistent spacing and separators

---

## Summary

### Key Design Principles

1. **Configuration-Driven**: All intents, enrichments, and data sources defined in YAML
2. **Separation of Concerns**: Each method has a single, clear responsibility
3. **Error Resilience**: Graceful degradation at every layer
4. **User-Friendly**: Clear output, helpful error messages
5. **Extensible**: Easy to add new intents without code changes

### Data Flow Summary

```
User Query
    ↓
[LLM Classification]
    ↓
Primary Intents
    ↓
[Enrichment Rules]
    ↓
Enriched Intents
    ↓
[Data Source Mapping]
    ↓
Required Data Sources
    ↓
Complete Classification Result
```

### Performance Characteristics

- **Initialization**: O(n) where n = number of intents (builds mappings)
- **Classification**: O(1) AWS API call + O(m) enrichment where m = number of primary intents (typically 1-3)
- **Data Source Lookup**: O(k) where k = number of enriched intents (typically 3-10)
- **Overall**: Dominated by AWS Bedrock latency (~500ms-2s)

### Extension Points

To extend the system:

1. **Add new intents**: Edit `intent_categories.yaml`
2. **Add enrichment rules**: Edit `enrichment_rules.yaml`
3. **Add data sources**: Edit `data_sources.yaml`
4. **Custom processing**: Override `classify()` method
5. **Alternative LLM**: Replace `_call_bedrock()` method

---

## Conclusion

The `intent_classifier.py` implementation provides a robust, maintainable, and extensible system for classifying user queries in an observability context. Its configuration-driven design allows for easy updates without code changes, while its error handling ensures reliable operation in production environments.
