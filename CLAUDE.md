# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an intent classification system for a Conversational SLO Manager. It maps natural language queries to specific intents and determines which data sources should be queried to answer them. The system is configuration-driven using YAML files and uses AWS Bedrock (Claude 3.5) for intelligent intent classification.

## Development Commands

**Setup**:
```bash
./setup.sh                           # Initial setup: create venv, install deps
source venv/bin/activate             # Activate virtual environment
pip install -r requirements.txt      # Install/update dependencies
```

**Running**:
```bash
python intent_classifier.py          # Interactive classification mode
python test_classifier.py            # Run comprehensive test suite
```

**Configuration**:
- AWS credentials must be configured in `.env` file (no .env.example provided in repo)
- Environment variables: `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BEDROCK_MODEL_ID`, `MAX_TOKENS`, `TEMPERATURE`
- Model access must be enabled in AWS Bedrock console for Anthropic Claude models

**Note**: There is a note in README.md: "CUSTOMER_IMPACT enrichement dalna he" - indicating planned work to add CUSTOMER_IMPACT enrichment

## Architecture

### Core Components

**Intent Hierarchy**: Two-level categorization system
- **Categories** (9 top-level): STATE, TREND, PATTERN, CAUSE, IMPACT, ACTION, PREDICT, OPTIMIZE, EVIDENCE
- **Intents** (30+ specific): Each category contains multiple specific intents (e.g., ROOT_CAUSE_SINGLE, BLAST_RADIUS)
- Each intent defines its own required data sources at the intent level

**Data Sources** (4 types):
- `java_stats_api`: REST API for real-time metrics, service health, performance data
- `postgres`: Relational database for SLOs, alerts, incidents, change history
- `clickhouse`: Analytical database for trends, patterns, historical analysis, AI memory
- `opensearch`: Search engine for logs, traces, error patterns, audit trails

**Enrichment System**: Automatic intent expansion with circular reference support
- When a primary intent is detected, related intents are automatically included
- Example: `ROOT_CAUSE_SINGLE` automatically enriches with `UNDERCURRENTS_TREND` and `MITIGATION_STEPS`
- System handles circular enrichment (e.g., `BLAST_RADIUS` → `USER_JOURNEY_IMPACT` → `BLAST_RADIUS`) by using sets
- This creates comprehensive answers by pulling relevant context

### Configuration Files

**intent_categories.yaml**:
- Defines all intent categories and their specific intents
- Each intent has: description, example queries, and required data sources
- Format: `CATEGORY > intent > {description, examples[], data_sources[]}`

**enrichment_rules.yaml**:
- Maps primary intents to related intents that should be auto-included
- Creates comprehensive query plans by combining related contexts
- Format: `PRIMARY_INTENT: [list of related intents]`

**data_sources.yaml**:
- Defines available data sources and their capabilities
- Includes connection settings (timeouts, pool sizes, retry attempts)
- Maps intent categories to their required data sources

## Key Design Patterns

**Intent Mapping Strategy**:
1. User query → Intent detection (from 30+ specific intents)
2. Enrichment rules apply → Related intents added automatically
3. Data sources determined → Based on all intents in the enriched set
4. Query execution → Multi-source data retrieval

**Data Source Selection Logic**:
- Each intent declares required data sources
- Enrichment can expand the data source list
- Categories also have default data source mappings for fallback

**Examples of Intent Chains**:
- `ALERT_DEBUG` → auto-includes `ROOT_CAUSE_SINGLE`, `BLAST_RADIUS`, `MITIGATION_STEPS`
- `SLO_STATUS` → auto-includes `SLO_BURN_TREND`, `RISK_PREDICTION`
- `PERFORMANCE_BOTTLENECK` → auto-includes `QUERY_OPTIMIZATION`, `RESOURCE_WASTE`

## Modification Guidelines

**Adding a new intent**:
1. Add to appropriate category in `intent_categories.yaml`
2. Include description, example queries, and data sources
3. Add enrichment rules in `enrichment_rules.yaml` if related to other intents
4. Verify data source capabilities match in `data_sources.yaml`

**Modifying enrichment rules**:
- Consider the query context: Does the enrichment add value?
- Circular enrichment references are allowed - system handles them using sets (e.g., BLAST_RADIUS ↔ USER_JOURNEY_IMPACT)
- Balance comprehensiveness vs. query performance
- Enrichment happens only one level deep (primary intents → enrichments, but enrichments don't trigger further enrichments)

**Data source changes**:
- Update capabilities list when data source features change
- Adjust timeouts based on query performance
- Maintain intent_data_sources mapping consistency with individual intent definitions

## Implementation

### Python Intent Classifier (`intent_classifier.py`)

**Main Class**: `IntentClassifier`
- Loads YAML configurations at initialization using `_load_yaml()`
- Builds intent-to-data-source mapping from configuration via `_build_intent_data_source_map()`
- Uses AWS Bedrock runtime client (`bedrock-runtime` service) for LLM inference
- Temperature set to 0.0 for deterministic classification (configurable via MAX_TOKENS env var, default: 10)

**Classification Flow**:
1. `classify(user_query)` → Calls AWS Bedrock with system prompt
2. LLM returns JSON array of primary intent(s) like `["INTENT1", "INTENT2"]`
3. `_get_enrichment_intents()` → Expands with related intents from enrichment rules (uses set to avoid duplicates)
4. `_get_data_sources()` → Collects all required data sources from enriched intent set (sorted alphabetically)
5. Returns complete classification result with enrichment details

**Return Format**:
```python
{
    "query": "User's original question",
    "primary_intents": ["INTENT1", "INTENT2"],           # From LLM
    "enriched_intents": ["INTENT1", "INTENT2", "..."],   # Primary + enrichments (sorted)
    "data_sources": ["datasource1", "datasource2"],      # Collected from all enriched intents
    "enrichment_details": {                              # Which enrichments came from which primary
        "INTENT1": ["ENRICHMENT1", "ENRICHMENT2"]
    }
}
```

**Error Handling**:
- On classification failure, returns dict with `"error"` key and empty arrays for all fields
- Gracefully handles JSON parsing errors by printing LLM response and returning empty array
- Catches AWS ClientError for credential/permission issues
- YAML loading errors are caught and empty dict returned (prints error message)

**Key Methods**:
- `_call_bedrock()`: Handles AWS Bedrock API calls using Messages API format (`anthropic_version: bedrock-2023-05-31`), robust JSON extraction (finds `[...]` in response), error handling for ClientError and JSONDecodeError
- `_build_system_prompt()`: Dynamically generates LLM prompt from intent_categories.yaml structure, includes descriptions and first example for each intent
- `_build_intent_data_source_map()`: Traverses YAML to create flat intent→data_sources dictionary
- `print_result()`: Pretty-prints classification results with emoji indicators (🎯 for primary, 🔄 for enrichment, 💾 for data sources)

### Additional Component: Confidence Scoring (`confidence.py`)

**Purpose**: Placeholder for future confidence scoring system (not currently integrated into main flow)

**Key Classes**:
- `ConfidenceLevel`: Enum with HIGH, MEDIUM, LOW, VERY_LOW levels
- `ConfidenceScorer`: Scores classifications based on heuristics (query length, vague terms, specific service names)

**Features** (for future integration):
- Configurable thresholds for confidence levels (default: high=0.85, medium=0.7, low=0.5)
- Clarification suggestions when confidence is low
- Fallback intent recommendations per category
- Historical confidence statistics tracking

### Setup and Testing

**Quick Setup**: Run `./setup.sh` to:
- Create Python virtual environment
- Install dependencies from `requirements.txt`
- Copy `.env.example` to `.env` (if not exists)

**Configuration**: Create `.env` file manually with:
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
MAX_TOKENS=10
TEMPERATURE=0.0
```

**Testing**:
- Interactive mode: `python intent_classifier.py` (enter queries one at a time, type 'quit' to exit)
- Automated tests: `python test_classifier.py` (runs 22 example queries covering all 9 categories)
- Test includes 0.5s delay between queries to avoid rate limiting

**Dependencies** (requirements.txt):
- `boto3>=1.34.0`: AWS SDK for Bedrock API calls
- `pyyaml>=6.0.1`: YAML configuration parsing
- `python-dotenv>=1.0.0`: Environment variable management
