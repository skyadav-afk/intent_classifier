# Shailendra_Intent_Analysis
CUSTOMER_IMPACT enrichement dalna he 
## Intent Classifier for Conversational SLO Manager

An intelligent intent classification system that uses AWS Bedrock to classify user queries into specific intents and determine the required data sources for answering observability and SRE questions.

## Features

- **Multi-Intent Classification**: Detects one or more intents from user queries
- **Automatic Enrichment**: Automatically includes related intents for comprehensive answers
- **Data Source Mapping**: Determines which data sources are needed to answer the query
- **AWS Bedrock Integration**: Uses Claude 3.5 Sonnet for accurate intent classification
- **Interactive CLI**: Test the classifier with real queries

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Bedrock

Copy the example environment file and configure your AWS credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your AWS credentials:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
MAX_TOKENS=500
TEMPERATURE=0.0
```

### 3. Ensure Model Access

Make sure you have access to Claude models in AWS Bedrock:
1. Go to AWS Console → Bedrock → Model Access
2. Request access to Anthropic Claude models
3. Wait for approval (usually instant)

## Usage

### Interactive Mode

Run the classifier in interactive mode:

```bash
python intent_classifier.py
```

Then enter your queries:

```
Query: Why is payment-api failing?
Query: What alerts are active and what incidents are open?
Query: Are we meeting our SLOs?
```

### Programmatic Usage

```python
from intent_classifier import IntentClassifier

# Initialize classifier
classifier = IntentClassifier()

# Classify a query
result = classifier.classify("Why is payment-api failing?")

# Access results
print(result['primary_intents'])      # ['ROOT_CAUSE_SINGLE']
print(result['enriched_intents'])     # ['ROOT_CAUSE_SINGLE', 'UNDERCURRENTS_TREND', 'MITIGATION_STEPS']
print(result['data_sources'])         # ['java_stats_api', 'opensearch', 'clickhouse']
print(result['enrichment_details'])   # {'ROOT_CAUSE_SINGLE': ['UNDERCURRENTS_TREND', 'MITIGATION_STEPS']}
```

## Example Queries

### Single Intent
- "How is my application now?" → `CURRENT_HEALTH`
- "Is payment-api healthy?" → `SERVICE_HEALTH`
- "Which SLOs are breached?" → `SLO_STATUS`

### Multiple Intents
- "What alerts are active and what incidents are open?" → `ALERT_STATUS`, `INCIDENT_STATUS`
- "Are we meeting our SLOs and what's the trend?" → `SLO_STATUS`, `SLO_BURN_TREND`

### With Enrichment
Query: "Why is payment-api failing?"
- Primary Intent: `ROOT_CAUSE_SINGLE`
- Enriched Intents: `UNDERCURRENTS_TREND`, `MITIGATION_STEPS`
- Data Sources: `java_stats_api`, `opensearch`, `clickhouse`

## Architecture

### Intent Categories

1. **STATE**: Current status and health
2. **TREND**: Changes over time
3. **PATTERN**: Recurring behaviors
4. **CAUSE**: Root cause analysis
5. **IMPACT**: Blast radius and business impact
6. **ACTION**: Decision support and mitigation
7. **PREDICT**: Future risk prediction
8. **OPTIMIZE**: Performance and cost tuning
9. **EVIDENCE**: Explainability and audit

### Data Sources

- **java_stats_api**: Real-time metrics and service health
- **postgres**: SLOs, alerts, incidents, change history
- **clickhouse**: Historical data, trends, patterns
- **opensearch**: Logs, traces, error patterns

### Enrichment Logic

The system automatically enriches queries with related intents. For example:
- `ROOT_CAUSE_SINGLE` → adds `UNDERCURRENTS_TREND`, `MITIGATION_STEPS`
- `ALERT_DEBUG` → adds `ROOT_CAUSE_SINGLE`, `BLAST_RADIUS`, `MITIGATION_STEPS`
- `SLO_STATUS` → adds `SLO_BURN_TREND`, `RISK_PREDICTION`

## Configuration Files

- `intent_categories.yaml`: Defines all intent categories and their intents
- `enrichment_rules.yaml`: Maps primary intents to related intents
- `data_sources.yaml`: Defines data sources and their capabilities

## Troubleshooting

### AWS Credentials Error
```
NoCredentialsError: Unable to locate credentials
```
→ Make sure `.env` file exists with valid AWS credentials

### Model Access Error
```
AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel
```
→ Request access to Claude models in AWS Bedrock console

### JSON Parsing Error
→ Check the LLM response in the error message. The model might need adjustment or the prompt might need refinement.

## Output Format

The classifier returns a dictionary with:

```python
{
    "query": "User's original question",
    "primary_intents": ["INTENT1", "INTENT2"],
    "enriched_intents": ["INTENT1", "INTENT2", "ENRICHMENT1", "ENRICHMENT2"],
    "data_sources": ["java_stats_api", "postgres"],
    "enrichment_details": {
        "INTENT1": ["ENRICHMENT1", "ENRICHMENT2"]
    }
}
```
