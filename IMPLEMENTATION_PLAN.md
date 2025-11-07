# Feature Store & ML Pipeline Implementation Plan
## Comprehensive Roadmap for Production-Grade Sentiment Analysis System

---

## Executive Summary

This document provides a detailed implementation plan to transform the current **Sentiment-Analysis** project from a basic scikit-learn pipeline into a production-grade ML system with feature store, orchestration, monitoring, and scalability capabilities, inspired by the **Data-Systems-for-Toxic-Comment-Classification** reference architecture.

### Current State vs. Target Architecture

| Aspect | Current State | Target Architecture |
|--------|---------------|---------------------|
| **Data Storage** | Local CSV files | Delta Lake + PostgreSQL + MinIO |
| **Feature Engineering** | TF-IDF in-memory | Feature Store with versioning |
| **Data Processing** | Pandas (single-node) | PySpark (batch) + Flink (streaming) |
| **Orchestration** | Manual execution | Apache Airflow DAGs |
| **Data Quality** | None | Great Expectations validation |
| **ML Versioning** | Manual joblib files | DVC + MLflow registry |
| **Monitoring** | None | Prometheus + Grafana + ELK |
| **Data Transformation** | Inline Python | dbt models |
| **Query Layer** | Direct file access | Trino federated queries |
| **Deployment** | Local scripts | Docker Compose infrastructure |

---

## Phase 1: Infrastructure Setup (Week 1-2)

### 1.1 Docker-Based Infrastructure

**Objective:** Create reproducible, containerized environment for all services

**Components to Deploy:**

```yaml
Services:
  - MinIO (S3-compatible object storage)
  - PostgreSQL (transactional database + feature store)
  - Apache Spark (batch processing)
  - Apache Airflow (orchestration)
  - MLflow (model registry)
  - Prometheus + Grafana (monitoring)
  - Hive Metastore (metadata management)
  - Trino (federated query engine)
```

**Implementation Steps:**

1. **Create `docker-compose.yml`**
   ```yaml
   version: '3.8'
   services:
     minio:
       image: minio/minio:latest
       ports: ["9000:9000", "9001:9001"]
       volumes: ["./data/minio:/data"]
       command: server /data --console-address ":9001"

     postgres:
       image: postgres:14
       environment:
         POSTGRES_DB: feature_store
         POSTGRES_USER: mlops
         POSTGRES_PASSWORD: ${DB_PASSWORD}
       ports: ["5432:5432"]
       volumes: ["./data/postgres:/var/lib/postgresql/data"]

     spark-master:
       image: bitnami/spark:3.4
       ports: ["8080:8080", "7077:7077"]
       environment:
         SPARK_MODE: master

     airflow-webserver:
       image: apache/airflow:2.7.0
       depends_on: [postgres]
       ports: ["8081:8080"]
       volumes:
         - ./airflow/dags:/opt/airflow/dags
         - ./airflow/logs:/opt/airflow/logs

     mlflow:
       image: ghcr.io/mlflow/mlflow:v2.8.0
       ports: ["5000:5000"]
       command: >
         mlflow server
         --backend-store-uri postgresql://mlops:${DB_PASSWORD}@postgres/mlflow
         --default-artifact-root s3://mlflow-artifacts/
         --host 0.0.0.0

     prometheus:
       image: prom/prometheus:latest
       ports: ["9090:9090"]
       volumes: ["./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"]

     grafana:
       image: grafana/grafana:latest
       ports: ["3000:3000"]
       volumes: ["./monitoring/grafana:/var/lib/grafana"]
   ```

2. **Create Environment Configuration**
   - `.env` file for secrets management
   - `configs/` directory for service configurations

3. **Network and Volume Setup**
   - Shared network for inter-service communication
   - Persistent volumes for data durability

**Deliverables:**
- [ ] `docker-compose.yml` with all services
- [ ] `.env.example` template
- [ ] `configs/` directory structure
- [ ] Infrastructure startup script: `scripts/start_infrastructure.sh`
- [ ] Health check script: `scripts/check_services.sh`

---

### 1.2 Project Restructuring

**Objective:** Modularize codebase for scalability and maintainability

**New Directory Structure:**

```
Sentiment-Analysis/
├── airflow/
│   ├── dags/
│   │   ├── batch_ingestion_dag.py
│   │   ├── feature_engineering_dag.py
│   │   ├── model_training_dag.py
│   │   └── model_deployment_dag.py
│   ├── plugins/
│   └── logs/
│
├── batch_processing/
│   ├── data_ingestion.py
│   ├── feature_extraction.py
│   └── spark_jobs/
│       ├── preprocess_reviews.py
│       └── compute_features.py
│
├── stream_processing/
│   ├── kafka_consumer.py
│   ├── flink_jobs/
│   │   └── realtime_sentiment.py
│   └── debezium/ (CDC configuration)
│
├── data_transformation/
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_reviews.sql
│   │   │   └── stg_features.sql
│   │   └── marts/
│   │       ├── fact_predictions.sql
│   │       └── dim_models.sql
│   ├── dbt_project.yml
│   └── profiles.yml
│
├── data_validation/
│   ├── expectations/
│   │   ├── raw_data_suite.json
│   │   └── features_suite.json
│   ├── checkpoints/
│   └── validate.py
│
├── feature_store/
│   ├── __init__.py
│   ├── feature_registry.py
│   ├── feature_definitions/
│   │   ├── text_features.py
│   │   ├── statistical_features.py
│   │   └── embedding_features.py
│   ├── storage_backends/
│   │   ├── delta_store.py
│   │   └── postgres_store.py
│   └── versioning.py
│
├── model_experiment/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── dataloader.py
│   ├── extract_data.py
│   └── notebooks/
│       └── advanced_eda.ipynb
│
├── configs/
│   ├── airflow.cfg
│   ├── spark-defaults.conf
│   ├── mlflow.yml
│   └── feature_store.yaml
│
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── ml_metrics.json
│   │       └── data_quality.json
│   └── alerts/
│       └── model_performance.yml
│
├── utils/
│   ├── database_utils.py
│   ├── s3_utils.py
│   ├── logging_config.py
│   └── config_loader.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── deltalake/
│   ├── minio/
│   └── postgres/
│
├── scripts/
│   ├── setup_database.py
│   ├── create_tables.sql
│   ├── seed_data.py
│   └── migrate_existing_data.py
│
├── docker-compose.yml
├── .env.example
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── pyproject.toml
└── README.md
```

**Implementation Steps:**

1. Create directory structure: `mkdir -p {airflow/dags,batch_processing,feature_store,...}`
2. Move existing code to appropriate locations
3. Create `__init__.py` files for Python packages
4. Update import paths throughout codebase

**Deliverables:**
- [ ] New directory structure created
- [ ] Existing code migrated and organized
- [ ] `README.md` updated with new structure
- [ ] Migration script: `scripts/migrate_project_structure.sh`

---

## Phase 2: Feature Store Implementation (Week 3-4)

### 2.1 Storage Layer Setup

**Objective:** Implement multi-tier feature storage with Delta Lake and PostgreSQL

**Architecture:**

```
Data Flow:
Raw Data → Delta Lake (Bronze) → Delta Lake (Silver) → PostgreSQL (Gold)
                                                     ↓
                                              Feature Serving API
```

**Implementation Steps:**

#### 2.1.1 Delta Lake Integration

**File:** `feature_store/storage_backends/delta_store.py`

```python
from delta import DeltaTable, configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import os

class DeltaFeatureStore:
    """Manages feature storage using Delta Lake format"""

    def __init__(self, base_path: str = "s3a://features/"):
        self.base_path = base_path
        self.spark = self._initialize_spark()

    def _initialize_spark(self) -> SparkSession:
        builder = SparkSession.builder \
            .appName("FeatureStore") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT")) \
            .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY")) \
            .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY")) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

        return configure_spark_with_delta_pip(builder).getOrCreate()

    def write_features(self, df, feature_group: str, mode: str = "overwrite"):
        """Write features to Delta Lake with versioning"""
        path = f"{self.base_path}/{feature_group}"
        df.write.format("delta").mode(mode).save(path)

    def read_features(self, feature_group: str, version: int = None):
        """Read features from Delta Lake (optionally at specific version)"""
        path = f"{self.base_path}/{feature_group}"
        if version:
            return self.spark.read.format("delta").option("versionAsOf", version).load(path)
        return self.spark.read.format("delta").load(path)

    def time_travel_features(self, feature_group: str, timestamp: str):
        """Query features as of a specific timestamp"""
        path = f"{self.base_path}/{feature_group}"
        return self.spark.read.format("delta").option("timestampAsOf", timestamp).load(path)

    def get_feature_history(self, feature_group: str):
        """Get version history for a feature group"""
        path = f"{self.base_path}/{feature_group}"
        delta_table = DeltaTable.forPath(self.spark, path)
        return delta_table.history()
```

#### 2.1.2 PostgreSQL Feature Store

**File:** `feature_store/storage_backends/postgres_store.py`

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

class PostgresFeatureStore:
    """Production feature store using PostgreSQL"""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.metadata = MetaData()
        self.Session = sessionmaker(bind=self.engine)

    def create_feature_table(self, table_name: str, schema: Dict[str, Any]):
        """Create a feature table with specified schema"""
        columns = [Column('id', Integer, primary_key=True, autoincrement=True)]
        columns.append(Column('feature_timestamp', DateTime, default=datetime.utcnow))
        columns.append(Column('entity_id', String(255), index=True))

        for col_name, col_type in schema.items():
            columns.append(Column(col_name, col_type))

        table = Table(table_name, self.metadata, *columns, schema='features')
        self.metadata.create_all(self.engine)
        return table

    def write_features(self, table_name: str, features_df: pd.DataFrame):
        """Write features to PostgreSQL table"""
        features_df['feature_timestamp'] = datetime.utcnow()
        features_df.to_sql(
            table_name,
            self.engine,
            schema='features',
            if_exists='append',
            index=False,
            method='multi'
        )

    def read_features(self, table_name: str, entity_ids: List[str] = None) -> pd.DataFrame:
        """Read latest features for specified entities"""
        query = f"""
            SELECT * FROM features.{table_name}
            WHERE feature_timestamp = (
                SELECT MAX(feature_timestamp)
                FROM features.{table_name}
            )
        """
        if entity_ids:
            entity_list = "', '".join(entity_ids)
            query += f" AND entity_id IN ('{entity_list}')"

        return pd.read_sql(query, self.engine)

    def get_feature_versions(self, table_name: str, entity_id: str) -> pd.DataFrame:
        """Get all versions of features for an entity"""
        query = f"""
            SELECT * FROM features.{table_name}
            WHERE entity_id = '{entity_id}'
            ORDER BY feature_timestamp DESC
        """
        return pd.read_sql(query, self.engine)
```

#### 2.1.3 Feature Registry

**File:** `feature_store/feature_registry.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import yaml

class FeatureType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"

@dataclass
class FeatureDefinition:
    """Defines a feature with metadata"""
    name: str
    feature_type: FeatureType
    description: str
    computation_fn: Callable
    dependencies: List[str]
    version: str
    owner: str
    tags: List[str]

class FeatureRegistry:
    """Central registry for all feature definitions"""

    def __init__(self, config_path: str = "configs/feature_store.yaml"):
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_groups: Dict[str, List[str]] = {}
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """Load feature configurations from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.feature_groups = config.get('feature_groups', {})

    def register_feature(self, feature_def: FeatureDefinition):
        """Register a new feature"""
        if feature_def.name in self.features:
            raise ValueError(f"Feature {feature_def.name} already registered")
        self.features[feature_def.name] = feature_def

    def get_feature(self, feature_name: str) -> FeatureDefinition:
        """Retrieve feature definition"""
        if feature_name not in self.features:
            raise KeyError(f"Feature {feature_name} not found in registry")
        return self.features[feature_name]

    def get_feature_group(self, group_name: str) -> List[FeatureDefinition]:
        """Get all features in a group"""
        if group_name not in self.feature_groups:
            raise KeyError(f"Feature group {group_name} not found")
        return [self.get_feature(f) for f in self.feature_groups[group_name]]

    def list_features(self, tags: List[str] = None) -> List[str]:
        """List all features, optionally filtered by tags"""
        if not tags:
            return list(self.features.keys())
        return [
            name for name, feat in self.features.items()
            if any(tag in feat.tags for tag in tags)
        ]
```

**Configuration File:** `configs/feature_store.yaml`

```yaml
storage:
  delta_lake:
    base_path: "s3a://features/"
    warehouse: "/opt/spark/warehouse"
  postgres:
    host: "postgres"
    port: 5432
    database: "feature_store"
    schema: "features"

feature_groups:
  text_statistics:
    - word_count
    - char_count
    - avg_word_length
    - sentence_count
    - stopword_ratio

  tfidf_features:
    - tfidf_vector
    - top_tfidf_terms

  sentiment_features:
    - sentiment_score
    - sentiment_magnitude
    - emotion_scores

  embedding_features:
    - bert_embeddings
    - word2vec_avg

versioning:
  strategy: "timestamp"  # or "semantic"
  retention_days: 90

quality:
  validation_enabled: true
  expectation_suite: "features_suite"
```

**Deliverables:**
- [ ] Delta Lake integration with MinIO
- [ ] PostgreSQL feature tables schema
- [ ] Feature Registry implementation
- [ ] Storage backend abstraction layer
- [ ] Configuration management
- [ ] Unit tests for storage backends

---

### 2.2 Feature Engineering Pipeline

**Objective:** Create modular, versioned feature computation pipeline

**File:** `feature_store/feature_definitions/text_features.py`

```python
import pandas as pd
import numpy as np
from typing import Dict, Any
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class TextFeatureExtractor:
    """Extract statistical and NLP features from text"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf = TfidfVectorizer(max_features=100)

    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic text statistics"""
        features = pd.DataFrame()

        features['word_count'] = df['review'].apply(lambda x: len(x.split()))
        features['char_count'] = df['review'].apply(len)
        features['avg_word_length'] = df['review'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        features['sentence_count'] = df['review'].apply(
            lambda x: len([s for s in x.split('.') if s.strip()])
        )
        features['capital_ratio'] = df['review'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        features['punctuation_count'] = df['review'].apply(
            lambda x: sum(1 for c in x if c in '!?.,;:')
        )

        return features

    def extract_tfidf_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Extract TF-IDF features"""
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(df['review'])
        else:
            tfidf_matrix = self.tfidf.transform(df['review'])

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        return tfidf_df

    def extract_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract linguistic features using spaCy"""
        features = pd.DataFrame()

        def analyze_text(text):
            doc = self.nlp(text)
            return {
                'noun_count': len([token for token in doc if token.pos_ == 'NOUN']),
                'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
                'adj_count': len([token for token in doc if token.pos_ == 'ADJ']),
                'entity_count': len(doc.ents),
            }

        linguistic_data = df['review'].apply(analyze_text)
        features = pd.DataFrame(linguistic_data.tolist())

        return features
```

**File:** `batch_processing/feature_extraction.py`

```python
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, length, split, size, avg
from pyspark.sql.types import FloatType, IntegerType
from feature_store.storage_backends.delta_store import DeltaFeatureStore
from feature_store.feature_registry import FeatureRegistry

class SparkFeaturePipeline:
    """Batch feature extraction using PySpark"""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.feature_store = DeltaFeatureStore()
        self.registry = FeatureRegistry()

    def compute_text_features(self, df: DataFrame) -> DataFrame:
        """Compute text features at scale"""

        # Basic text statistics
        df = df.withColumn('word_count', size(split(col('review'), ' ')))
        df = df.withColumn('char_count', length(col('review')))
        df = df.withColumn('avg_word_length',
                          length(col('review')) / col('word_count'))

        # Custom UDFs for complex features
        @udf(returnType=IntegerType())
        def count_punctuation(text):
            return sum(1 for c in text if c in '!?.,;:')

        @udf(returnType=FloatType())
        def capital_ratio(text):
            if len(text) == 0:
                return 0.0
            return sum(1 for c in text if c.isupper()) / len(text)

        df = df.withColumn('punctuation_count', count_punctuation(col('review')))
        df = df.withColumn('capital_ratio', capital_ratio(col('review')))

        return df

    def save_to_feature_store(self, df: DataFrame, feature_group: str):
        """Save computed features to Delta Lake"""
        self.feature_store.write_features(df, feature_group, mode="overwrite")
        print(f"✓ Features saved to feature group: {feature_group}")
```

**Deliverables:**
- [ ] Text feature extractors (statistical, TF-IDF, linguistic)
- [ ] Embedding feature extractors (Word2Vec, BERT)
- [ ] PySpark batch feature computation
- [ ] Feature versioning logic
- [ ] Feature documentation generation

---

### 2.3 Data Validation with Great Expectations

**Objective:** Ensure feature quality before storage

**File:** `data_validation/validate.py`

```python
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import Checkpoint

class FeatureValidator:
    """Validate features using Great Expectations"""

    def __init__(self, context_root_dir: str = "data_validation/"):
        self.context = gx.get_context(context_root_dir=context_root_dir)

    def create_expectation_suite(self, suite_name: str):
        """Create a new expectation suite"""
        suite = self.context.add_expectation_suite(suite_name)

        # Add expectations
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="word_count",
                min_value=1,
                max_value=5000
            )
        )

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="tfidf_vector"
            )
        )

        suite.add_expectation(
            gx.expectations.ExpectColumnMeanToBeBetween(
                column="sentiment_score",
                min_value=-1.0,
                max_value=1.0
            )
        )

        self.context.save_expectation_suite(suite)
        return suite

    def validate_features(self, df, suite_name: str) -> bool:
        """Validate a dataframe against expectations"""

        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name="features",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default"}
        )

        checkpoint_config = {
            "name": "feature_checkpoint",
            "config_version": 1,
            "class_name": "Checkpoint",
            "validations": [
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": suite_name
                }
            ]
        }

        checkpoint = Checkpoint(**checkpoint_config)
        result = self.context.run_checkpoint(checkpoint=checkpoint)

        return result["success"]
```

**Configuration:** `data_validation/expectations/features_suite.json`

```json
{
  "expectation_suite_name": "features_quality",
  "expectations": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {"column": "review"}
    },
    {
      "expectation_type": "expect_column_values_to_be_between",
      "kwargs": {
        "column": "word_count",
        "min_value": 1,
        "max_value": 5000
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_of_type",
      "kwargs": {
        "column": "sentiment_score",
        "type_": "float"
      }
    },
    {
      "expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {
        "column": "capital_ratio",
        "min_value": 0.0,
        "max_value": 1.0
      }
    }
  ]
}
```

**Deliverables:**
- [ ] Great Expectations configuration
- [ ] Expectation suites for features
- [ ] Validation checkpoints
- [ ] Integration with feature pipeline
- [ ] Validation reports and alerts

---

## Phase 3: Data Transformation with dbt (Week 5)

### 3.1 dbt Project Setup

**Objective:** Create SQL-based transformation layer for feature curation

**File:** `data_transformation/dbt_project.yml`

```yaml
name: 'sentiment_features'
version: '1.0.0'
config-version: 2

profile: 'feature_store'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

clean-targets:
  - "target"
  - "dbt_packages"

models:
  sentiment_features:
    staging:
      +materialized: view
      +schema: staging
    marts:
      +materialized: table
      +schema: production
```

**File:** `data_transformation/profiles.yml`

```yaml
feature_store:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: mlops
      password: "{{ env_var('DB_PASSWORD') }}"
      dbname: feature_store
      schema: staging
      threads: 4

    prod:
      type: postgres
      host: postgres
      port: 5432
      user: mlops
      password: "{{ env_var('DB_PASSWORD') }}"
      dbname: feature_store
      schema: production
      threads: 8
```

### 3.2 dbt Models

**File:** `data_transformation/models/staging/stg_raw_reviews.sql`

```sql
-- Staging model: Clean and standardize raw reviews
{{ config(materialized='view') }}

WITH source AS (
    SELECT * FROM {{ source('raw', 'reviews') }}
),

cleaned AS (
    SELECT
        id AS review_id,
        review_text,
        sentiment AS sentiment_label,
        LOWER(TRIM(review_text)) AS cleaned_text,
        LENGTH(review_text) AS text_length,
        CURRENT_TIMESTAMP AS processed_at
    FROM source
    WHERE review_text IS NOT NULL
      AND LENGTH(review_text) > 10  -- Filter out too short reviews
)

SELECT * FROM cleaned
```

**File:** `data_transformation/models/staging/stg_text_features.sql`

```sql
-- Staging model: Text statistical features
{{ config(materialized='view') }}

WITH reviews AS (
    SELECT * FROM {{ ref('stg_raw_reviews') }}
),

features AS (
    SELECT
        review_id,
        cleaned_text,
        sentiment_label,

        -- Text statistics
        LENGTH(cleaned_text) AS char_count,
        LENGTH(cleaned_text) - LENGTH(REPLACE(cleaned_text, ' ', '')) + 1 AS word_count,
        LENGTH(REGEXP_REPLACE(cleaned_text, '[^!?.,;:]', '', 'g')) AS punctuation_count,

        -- Derived metrics
        ROUND(
            CAST(LENGTH(cleaned_text) AS NUMERIC) /
            NULLIF(LENGTH(cleaned_text) - LENGTH(REPLACE(cleaned_text, ' ', '')) + 1, 0),
            2
        ) AS avg_word_length,

        processed_at
    FROM reviews
)

SELECT * FROM features
```

**File:** `data_transformation/models/marts/fact_training_data.sql`

```sql
-- Production model: Training dataset with all features
{{ config(
    materialized='table',
    indexes=[{'columns': ['review_id'], 'unique': True}]
) }}

WITH text_features AS (
    SELECT * FROM {{ ref('stg_text_features') }}
),

tfidf_features AS (
    SELECT * FROM {{ source('features', 'tfidf_features') }}
),

embeddings AS (
    SELECT * FROM {{ source('features', 'embedding_features') }}
),

joined AS (
    SELECT
        t.review_id,
        t.cleaned_text,
        t.sentiment_label,
        t.char_count,
        t.word_count,
        t.punctuation_count,
        t.avg_word_length,
        tf.tfidf_vector,
        e.bert_embedding,
        t.processed_at AS feature_timestamp
    FROM text_features t
    LEFT JOIN tfidf_features tf ON t.review_id = tf.review_id
    LEFT JOIN embeddings e ON t.review_id = e.review_id
)

SELECT * FROM joined
```

**File:** `data_transformation/models/marts/dim_feature_metadata.sql`

```sql
-- Dimension table: Feature metadata and statistics
{{ config(materialized='table') }}

WITH feature_stats AS (
    SELECT
        'word_count' AS feature_name,
        'numerical' AS feature_type,
        AVG(word_count) AS mean_value,
        STDDEV(word_count) AS std_value,
        MIN(word_count) AS min_value,
        MAX(word_count) AS max_value,
        CURRENT_TIMESTAMP AS computed_at
    FROM {{ ref('stg_text_features') }}

    UNION ALL

    SELECT
        'char_count' AS feature_name,
        'numerical' AS feature_type,
        AVG(char_count) AS mean_value,
        STDDEV(char_count) AS std_value,
        MIN(char_count) AS min_value,
        MAX(char_count) AS max_value,
        CURRENT_TIMESTAMP AS computed_at
    FROM {{ ref('stg_text_features') }}
)

SELECT * FROM feature_stats
```

### 3.3 dbt Tests

**File:** `data_transformation/models/staging/schema.yml`

```yaml
version: 2

models:
  - name: stg_raw_reviews
    description: "Cleaned and standardized raw reviews"
    columns:
      - name: review_id
        description: "Unique identifier for each review"
        tests:
          - unique
          - not_null

      - name: cleaned_text
        description: "Preprocessed review text"
        tests:
          - not_null

      - name: sentiment_label
        description: "Sentiment classification (positive/negative)"
        tests:
          - not_null
          - accepted_values:
              values: ['positive', 'negative']

  - name: stg_text_features
    description: "Statistical text features"
    columns:
      - name: word_count
        description: "Number of words in review"
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 1"

      - name: avg_word_length
        description: "Average word length"
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 1.0"
```

**Deliverables:**
- [ ] dbt project structure
- [ ] Staging models for data cleaning
- [ ] Mart models for production features
- [ ] Data quality tests
- [ ] Documentation with dbt docs
- [ ] Integration with Airflow for scheduling

---

## Phase 4: ML Pipeline with MLflow & DVC (Week 6-7)

### 4.1 DVC Setup for Versioning

**Objective:** Version datasets, features, and models

**File:** `dvc.yaml`

```yaml
stages:
  extract_data:
    cmd: python model_experiment/extract_data.py
    deps:
      - model_experiment/extract_data.py
      - configs/feature_store.yaml
    params:
      - extract.data_source
      - extract.date_range
    outs:
      - data/processed/training_data.csv

  feature_engineering:
    cmd: python batch_processing/feature_extraction.py
    deps:
      - data/processed/training_data.csv
      - batch_processing/feature_extraction.py
      - feature_store/feature_definitions/
    params:
      - features.text_features
      - features.embedding_model
    outs:
      - data/processed/features.parquet

  train_model:
    cmd: python model_experiment/train.py
    deps:
      - data/processed/features.parquet
      - model_experiment/train.py
      - model_experiment/model.py
    params:
      - train.model_type
      - train.hyperparameters
    metrics:
      - metrics/train_metrics.json:
          cache: false
    outs:
      - weights/model.pth

  evaluate_model:
    cmd: python model_experiment/evaluate.py
    deps:
      - weights/model.pth
      - data/processed/features.parquet
      - model_experiment/evaluate.py
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.png
      - plots/roc_curve.png
```

**File:** `.dvc/config`

```ini
[core]
    remote = minio

['remote "minio"']
    url = s3://dvc-storage/
    endpointurl = http://localhost:9000
    access_key_id = ${MINIO_ACCESS_KEY}
    secret_access_key = ${MINIO_SECRET_KEY}
```

**Commands:**

```bash
# Initialize DVC
dvc init

# Add data to tracking
dvc add data/IMDB-Dataset.csv

# Configure remote storage
dvc remote add -d minio s3://dvc-storage
dvc remote modify minio endpointurl http://localhost:9000

# Run pipeline
dvc repro

# Compare experiments
dvc exp show
dvc metrics diff
```

### 4.2 MLflow Tracking and Registry

**File:** `model_experiment/train.py`

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import SentimentDataset
from model import SentimentClassifier
import yaml
import json
from datetime import datetime

class MLflowTrainingPipeline:
    """Training pipeline with MLflow tracking"""

    def __init__(self, config_path: str = "configs/training.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        """Main training loop with MLflow tracking"""

        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Log parameters
            mlflow.log_params(self.config['train']['hyperparameters'])
            mlflow.log_param("model_type", self.config['train']['model_type'])
            mlflow.log_param("device", str(self.device))

            # Load data
            train_dataset = SentimentDataset(
                data_path=self.config['data']['train_path'],
                feature_columns=self.config['features']['text_features']
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['train']['hyperparameters']['batch_size'],
                shuffle=True
            )

            # Initialize model
            self.model = SentimentClassifier(
                input_dim=self.config['model']['input_dim'],
                hidden_dim=self.config['model']['hidden_dim'],
                output_dim=2  # binary classification
            ).to(self.device)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['train']['hyperparameters']['learning_rate']
            )

            # Training loop
            num_epochs = self.config['train']['hyperparameters']['num_epochs']
            for epoch in range(num_epochs):
                train_loss, train_acc = self._train_epoch(
                    train_loader, criterion, optimizer
                )

                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)

                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            # Save model
            model_path = f"weights/{self.config['train']['model_type']}_model.pth"
            torch.save(self.model.state_dict(), model_path)

            # Log model to MLflow
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name=self.config['mlflow']['model_name']
            )

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact("configs/training.yaml")

            # Log feature importance if available
            self._log_feature_importance()

            # Set model tags
            mlflow.set_tags({
                "model_type": self.config['train']['model_type'],
                "framework": "pytorch",
                "stage": "training"
            })

            return mlflow.active_run().info.run_id

    def _train_epoch(self, loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(loader):
            features, labels = features.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _log_feature_importance(self):
        """Log feature importance to MLflow"""
        # Extract feature importance from model (if applicable)
        # For neural networks, you might use attention weights or SHAP values
        pass
```

**File:** `model_experiment/evaluate.py`

```python
import mlflow
from mlflow.tracking import MlflowClient
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

class ModelEvaluator:
    """Evaluate model and log metrics to MLflow"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = MlflowClient()

    def evaluate(self, model, test_loader, device):
        """Comprehensive model evaluation"""

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        metrics = {
            "accuracy": (all_preds == all_labels).mean(),
            "roc_auc": roc_auc_score(all_labels, all_probs[:, 1])
        }

        # Classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        metrics.update({
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score']
        })

        # Log metrics to MLflow
        with mlflow.start_run(run_id=self.run_id):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)

        # Save metrics locally for DVC
        with open('metrics/eval_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Generate and save plots
        self._plot_confusion_matrix(all_labels, all_preds)
        self._plot_roc_curve(all_labels, all_probs[:, 1])

        return metrics

    def _plot_confusion_matrix(self, labels, preds):
        """Generate confusion matrix plot"""
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('plots/confusion_matrix.png')

        # Log to MLflow
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact('plots/confusion_matrix.png')

        plt.close()

    def _plot_roc_curve(self, labels, probs):
        """Generate ROC curve"""
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(labels, probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('plots/roc_curve.png')

        # Log to MLflow
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact('plots/roc_curve.png')

        plt.close()
```

### 4.3 Model Registry and Deployment

**File:** `model_experiment/register_model.py`

```python
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

class ModelRegistry:
    """Manage model lifecycle in MLflow Registry"""

    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def promote_model(self, model_name: str, run_id: str, stage: str = "Staging"):
        """Promote a model to a specific stage"""

        # Get model version
        model_uri = f"runs:/{run_id}/model"
        model_details = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        # Transition to stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage=stage
        )

        print(f"✓ Model version {model_details.version} promoted to {stage}")
        return model_details.version

    def get_production_model(self, model_name: str):
        """Get latest production model"""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"No production model found for {model_name}")
        return versions[0]

    def compare_models(self, model_name: str, metric: str = "eval_accuracy"):
        """Compare all model versions by metric"""
        versions = self.client.search_model_versions(f"name='{model_name}'")

        results = []
        for version in versions:
            run = self.client.get_run(version.run_id)
            metric_value = run.data.metrics.get(metric, None)
            results.append({
                "version": version.version,
                "stage": version.current_stage,
                "metric": metric_value,
                "run_id": version.run_id
            })

        return sorted(results, key=lambda x: x['metric'] or 0, reverse=True)
```

**Deliverables:**
- [ ] DVC pipeline configuration
- [ ] MLflow experiment tracking
- [ ] Model training with versioning
- [ ] Comprehensive evaluation metrics
- [ ] Model registry management
- [ ] Automated model comparison

---

## Phase 5: Orchestration with Apache Airflow (Week 8)

### 5.1 Airflow DAG for Batch Processing

**File:** `airflow/dags/batch_ingestion_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow/dags')

from utils.database_utils import check_database_connection
from utils.s3_utils import upload_to_s3

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'batch_data_ingestion',
    default_args=default_args,
    description='Ingest and process batch data to feature store',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['batch', 'feature-store']
)

# Task 1: Check prerequisites
check_db = PythonOperator(
    task_id='check_database_connection',
    python_callable=check_database_connection,
    dag=dag
)

# Task 2: Extract raw data
extract_data = BashOperator(
    task_id='extract_raw_data',
    bash_command='python /opt/airflow/dags/batch_processing/data_ingestion.py',
    dag=dag
)

# Task 3: Data preprocessing with Spark
preprocess_spark = SparkSubmitOperator(
    task_id='preprocess_with_spark',
    application='/opt/airflow/dags/batch_processing/spark_jobs/preprocess_reviews.py',
    name='preprocess_reviews',
    conn_id='spark_default',
    conf={
        'spark.executor.memory': '4g',
        'spark.executor.cores': '2',
        'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension'
    },
    dag=dag
)

# Task 4: Compute features
compute_features = SparkSubmitOperator(
    task_id='compute_features',
    application='/opt/airflow/dags/batch_processing/spark_jobs/compute_features.py',
    name='compute_features',
    conn_id='spark_default',
    dag=dag
)

# Task 5: Validate features
validate_features = PythonOperator(
    task_id='validate_features',
    python_callable=lambda: validate_with_great_expectations('features_suite'),
    dag=dag
)

# Task 6: Run dbt transformations
dbt_run = BashOperator(
    task_id='dbt_transformations',
    bash_command='cd /opt/airflow/dags/data_transformation && dbt run --profiles-dir .',
    dag=dag
)

# Task 7: dbt tests
dbt_test = BashOperator(
    task_id='dbt_tests',
    bash_command='cd /opt/airflow/dags/data_transformation && dbt test --profiles-dir .',
    dag=dag
)

# Task 8: Upload artifacts to S3
upload_artifacts = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    op_kwargs={
        'local_path': '/tmp/processed_features',
        'bucket': 'feature-artifacts',
        'prefix': '{{ ds }}'  # Date stamp
    },
    dag=dag
)

# Define task dependencies
check_db >> extract_data >> preprocess_spark >> compute_features
compute_features >> validate_features >> dbt_run >> dbt_test >> upload_artifacts
```

### 5.2 Model Training DAG

**File:** `airflow/dags/model_training_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Train and evaluate sentiment models',
    schedule_interval='0 4 * * 0',  # Weekly on Sunday at 4 AM
    catchup=False,
    tags=['ml', 'training']
)

# Task 1: Check if retraining is needed
def check_retraining_needed(**context):
    """Check data drift or performance degradation"""
    # Logic to compare current model performance vs. new data
    # Return task_id based on decision
    needs_retraining = True  # Implement actual check
    if needs_retraining:
        return 'extract_training_data'
    else:
        return 'skip_training'

check_retraining = BranchPythonOperator(
    task_id='check_retraining_needed',
    python_callable=check_retraining_needed,
    dag=dag
)

# Task 2: Extract training data from feature store
extract_data = BashOperator(
    task_id='extract_training_data',
    bash_command='dvc repro extract_data',
    dag=dag
)

# Task 3: Feature engineering
feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command='dvc repro feature_engineering',
    dag=dag
)

# Task 4: Model training
train_model = BashOperator(
    task_id='train_model',
    bash_command='dvc repro train_model',
    dag=dag
)

# Task 5: Model evaluation
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='dvc repro evaluate_model',
    dag=dag
)

# Task 6: Compare with production model
def compare_models(**context):
    """Compare new model vs. production model"""
    from model_experiment.register_model import ModelRegistry

    registry = ModelRegistry(tracking_uri="http://mlflow:5000")
    comparison = registry.compare_models("sentiment_classifier")

    # Get new model metrics from context
    new_accuracy = context['ti'].xcom_pull(task_ids='evaluate_model', key='accuracy')
    prod_accuracy = comparison[0]['metric']

    if new_accuracy > prod_accuracy:
        return 'promote_to_staging'
    else:
        return 'notify_team'

compare = BranchPythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    dag=dag
)

# Task 7: Promote to staging
def promote_model(**context):
    from model_experiment.register_model import ModelRegistry

    registry = ModelRegistry(tracking_uri="http://mlflow:5000")
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')
    registry.promote_model("sentiment_classifier", run_id, stage="Staging")

promote_staging = PythonOperator(
    task_id='promote_to_staging',
    python_callable=promote_model,
    dag=dag
)

# Task 8: Notify team
notify = BashOperator(
    task_id='notify_team',
    bash_command='echo "Model training completed. Check metrics at http://mlflow:5000"',
    dag=dag
)

# Task 9: Skip training
skip = BashOperator(
    task_id='skip_training',
    bash_command='echo "No retraining needed based on current metrics"',
    dag=dag
)

# Define dependencies
check_retraining >> [extract_data, skip]
extract_data >> feature_engineering >> train_model >> evaluate_model >> compare
compare >> [promote_staging, notify]
```

### 5.3 Monitoring DAG

**File:** `airflow/dags/monitoring_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

dag = DAG(
    'data_quality_monitoring',
    default_args=default_args,
    description='Monitor data quality and model performance',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    tags=['monitoring', 'quality']
)

def check_feature_drift():
    """Check for feature drift"""
    # Implement statistical tests for drift detection
    from scipy import stats
    import pandas as pd

    # Compare current features with historical baseline
    current_features = pd.read_parquet('data/processed/features.parquet')
    baseline_features = pd.read_parquet('data/baseline/features.parquet')

    # Kolmogorov-Smirnov test
    drift_detected = False
    for column in current_features.columns:
        if column in baseline_features.columns:
            statistic, pvalue = stats.ks_2samp(
                current_features[column],
                baseline_features[column]
            )
            if pvalue < 0.05:
                print(f"⚠️  Drift detected in feature: {column}")
                drift_detected = True

    return drift_detected

def check_model_performance():
    """Monitor model performance in production"""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri="http://mlflow:5000")

    # Get recent predictions and actual labels
    # Calculate online metrics
    # Compare with baseline threshold

    accuracy_threshold = 0.85
    current_accuracy = 0.88  # Fetch from monitoring database

    if current_accuracy < accuracy_threshold:
        print(f"⚠️  Model performance degraded: {current_accuracy:.3f} < {accuracy_threshold}")
        # Trigger alert
        return False
    return True

def check_data_freshness():
    """Check if data pipelines are running on schedule"""
    import psycopg2
    from datetime import datetime, timedelta

    conn = psycopg2.connect(
        host="postgres",
        database="feature_store",
        user="mlops",
        password=os.getenv("DB_PASSWORD")
    )

    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(feature_timestamp)
        FROM features.fact_training_data
    """)
    last_update = cursor.fetchone()[0]

    if datetime.now() - last_update > timedelta(days=1):
        print("⚠️  Data is stale - last update:", last_update)
        return False
    return True

# Define tasks
drift_check = PythonOperator(
    task_id='check_feature_drift',
    python_callable=check_feature_drift,
    dag=dag
)

performance_check = PythonOperator(
    task_id='check_model_performance',
    python_callable=check_model_performance,
    dag=dag
)

freshness_check = PythonOperator(
    task_id='check_data_freshness',
    python_callable=check_data_freshness,
    dag=dag
)

# Run checks in parallel
[drift_check, performance_check, freshness_check]
```

**Deliverables:**
- [ ] Batch ingestion DAG with Spark jobs
- [ ] Model training and evaluation DAG
- [ ] dbt integration in Airflow
- [ ] Monitoring and alerting DAG
- [ ] Custom Airflow operators for ML tasks
- [ ] DAG documentation and testing

---

## Phase 6: Monitoring & Observability (Week 9)

### 6.1 Prometheus Metrics

**File:** `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Airflow metrics
  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow-webserver:8080']
    metrics_path: '/admin/metrics'

  # MLflow metrics
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']

  # PostgreSQL exporter
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # MinIO metrics
  - job_name: 'minio'
    metrics_path: '/minio/v2/metrics/cluster'
    static_configs:
      - targets: ['minio:9000']

  # Custom ML metrics
  - job_name: 'ml_metrics'
    static_configs:
      - targets: ['ml-metrics-exporter:8000']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/*.yml'
```

**File:** `monitoring/alerts/model_performance.yml`

```yaml
groups:
  - name: ml_alerts
    interval: 5m
    rules:
      - alert: ModelAccuracyDegraded
        expr: model_accuracy < 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model accuracy below threshold"
          description: "Model {{ $labels.model_name }} accuracy is {{ $value }}"

      - alert: FeatureDriftDetected
        expr: feature_drift_score > 0.1
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Feature drift detected"
          description: "Drift in feature {{ $labels.feature_name }}: {{ $value }}"

      - alert: DataPipelineFailure
        expr: airflow_dag_failed > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Airflow DAG failed"
          description: "DAG {{ $labels.dag_id }} failed"

      - alert: FeatureStoreUnavailable
        expr: up{job="postgres"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Feature store unavailable"
          description: "PostgreSQL feature store is down"
```

### 6.2 Grafana Dashboards

**File:** `monitoring/grafana/dashboards/ml_metrics.json`

```json
{
  "dashboard": {
    "title": "ML Pipeline Metrics",
    "panels": [
      {
        "title": "Model Accuracy Over Time",
        "type": "graph",
        "targets": [
          {
            "expr": "model_accuracy{model_name='sentiment_classifier'}",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "title": "Feature Drift Scores",
        "type": "heatmap",
        "targets": [
          {
            "expr": "feature_drift_score",
            "legendFormat": "{{ feature_name }}"
          }
        ]
      },
      {
        "title": "Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, prediction_latency_bucket)",
            "legendFormat": "p95 latency"
          }
        ]
      },
      {
        "title": "Data Pipeline Status",
        "type": "stat",
        "targets": [
          {
            "expr": "airflow_dag_status{dag_id='batch_data_ingestion'}",
            "legendFormat": "Status"
          }
        ]
      },
      {
        "title": "Feature Store Query Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(postgres_queries_total[5m])",
            "legendFormat": "Queries/sec"
          }
        ]
      },
      {
        "title": "Model Registry Activity",
        "type": "table",
        "targets": [
          {
            "expr": "mlflow_model_versions",
            "legendFormat": "{{ model_name }}"
          }
        ]
      }
    ]
  }
}
```

### 6.3 Custom Metrics Exporter

**File:** `monitoring/ml_metrics_exporter.py`

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import psycopg2
import os
from mlflow.tracking import MlflowClient

# Define Prometheus metrics
model_accuracy = Gauge('model_accuracy', 'Current model accuracy', ['model_name'])
feature_drift = Gauge('feature_drift_score', 'Feature drift score', ['feature_name'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
predictions_total = Counter('predictions_total', 'Total predictions', ['model_name', 'prediction'])
feature_store_queries = Counter('feature_store_queries_total', 'Feature store queries')

class MLMetricsExporter:
    """Export ML metrics to Prometheus"""

    def __init__(self):
        self.mlflow_client = MlflowClient(tracking_uri="http://mlflow:5000")
        self.db_conn = psycopg2.connect(
            host="postgres",
            database="feature_store",
            user="mlops",
            password=os.getenv("DB_PASSWORD")
        )

    def collect_model_metrics(self):
        """Collect metrics from MLflow"""
        try:
            # Get production model
            versions = self.mlflow_client.get_latest_versions(
                "sentiment_classifier",
                stages=["Production"]
            )

            if versions:
                version = versions[0]
                run = self.mlflow_client.get_run(version.run_id)

                # Export accuracy metric
                accuracy = run.data.metrics.get('eval_accuracy', 0)
                model_accuracy.labels(model_name='sentiment_classifier').set(accuracy)

        except Exception as e:
            print(f"Error collecting model metrics: {e}")

    def collect_drift_metrics(self):
        """Collect feature drift scores from database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT feature_name, drift_score
                FROM monitoring.feature_drift
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 100
            """)

            for feature_name, drift_score in cursor.fetchall():
                feature_drift.labels(feature_name=feature_name).set(drift_score)

        except Exception as e:
            print(f"Error collecting drift metrics: {e}")

    def run(self, interval=30):
        """Run metrics collection loop"""
        start_http_server(8000)
        print("📊 Metrics exporter started on :8000")

        while True:
            self.collect_model_metrics()
            self.collect_drift_metrics()
            time.sleep(interval)

if __name__ == '__main__':
    exporter = MLMetricsExporter()
    exporter.run()
```

**Deliverables:**
- [ ] Prometheus configuration
- [ ] Grafana dashboards for ML metrics
- [ ] Custom metrics exporter
- [ ] Alert rules for model performance
- [ ] Integration with Airflow for DAG monitoring
- [ ] Logging infrastructure (ELK stack optional)

---

## Phase 7: Stream Processing (Optional - Week 10-11)

### 7.1 Kafka Setup for Real-time Data

**File:** `docker-compose.yml` (additions)

```yaml
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092

  debezium:
    image: debezium/connect:latest
    ports: ["8083:8083"]
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      CONFIG_STORAGE_TOPIC: debezium_configs
      OFFSET_STORAGE_TOPIC: debezium_offsets
```

### 7.2 CDC with Debezium

**File:** `stream_processing/debezium/postgres_connector.json`

```json
{
  "name": "postgres-feature-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "mlops",
    "database.password": "${DB_PASSWORD}",
    "database.dbname": "feature_store",
    "database.server.name": "feature_store",
    "table.include.list": "features.fact_training_data",
    "plugin.name": "pgoutput",
    "transforms": "route",
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
    "transforms.route.replacement": "$3"
  }
}
```

### 7.3 Apache Flink Stream Processing

**File:** `stream_processing/flink_jobs/realtime_sentiment.py`

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
import json

def process_review_stream():
    """Real-time sentiment analysis on streaming reviews"""

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)

    # Kafka consumer
    kafka_consumer = FlinkKafkaConsumer(
        topics='reviews',
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': 'kafka:9092',
            'group.id': 'sentiment-processor'
        }
    )

    # Define processing
    stream = env.add_source(kafka_consumer)

    # Transform and predict
    predictions = stream.map(lambda review: predict_sentiment(json.loads(review)))

    # Kafka producer
    kafka_producer = FlinkKafkaProducer(
        topic='predictions',
        serialization_schema=SimpleStringSchema(),
        producer_config={
            'bootstrap.servers': 'kafka:9092'
        }
    )

    predictions.add_sink(kafka_producer)

    env.execute("Realtime Sentiment Analysis")

def predict_sentiment(review_data):
    """Load model and predict sentiment"""
    # Load model from MLflow
    # Extract features from review
    # Return prediction
    pass

if __name__ == '__main__':
    process_review_stream()
```

**Deliverables:**
- [ ] Kafka cluster setup
- [ ] Debezium CDC connectors
- [ ] Apache Flink streaming jobs
- [ ] Real-time feature computation
- [ ] Stream-to-batch synchronization

---

## Phase 8: Advanced Features & Optimization (Week 12+)

### 8.1 Advanced Feature Engineering

**Implementations:**

1. **Word Embeddings (Word2Vec, GloVe)**
   ```python
   from gensim.models import Word2Vec

   def train_word2vec(texts):
       sentences = [text.split() for text in texts]
       model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
       return model
   ```

2. **Transformer Embeddings (BERT)**
   ```python
   from transformers import BertTokenizer, BertModel

   def extract_bert_embeddings(texts):
       tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       model = BertModel.from_pretrained('bert-base-uncased')

       embeddings = []
       for text in texts:
           inputs = tokenizer(text, return_tensors='pt', truncation=True)
           outputs = model(**inputs)
           embeddings.append(outputs.last_hidden_state.mean(dim=1))

       return embeddings
   ```

3. **Sentiment Lexicons**
4. **N-gram Features**
5. **Topic Modeling (LDA)**

### 8.2 Hyperparameter Optimization

**File:** `model_experiment/hyperparameter_tuning.py`

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    """Optuna objective function"""

    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Train model with these hyperparameters
    accuracy = train_and_evaluate(lr, hidden_dim, dropout)

    return accuracy

# Run optimization
mlflow_callback = MLflowCallback(tracking_uri="http://mlflow:5000")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])
```

### 8.3 Model Serving API

**File:** `api/serve.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyflink
from feature_store.feature_registry import FeatureRegistry

app = FastAPI(title="Sentiment Analysis API")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    prediction_id: str

# Load model at startup
model = mlflow.pytorch.load_model("models:/sentiment_classifier/Production")
feature_registry = FeatureRegistry()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Real-time sentiment prediction"""

    # Extract features
    features = feature_registry.extract_features(request.text)

    # Predict
    prediction = model.predict(features)

    return PredictionResponse(
        sentiment="positive" if prediction > 0.5 else "negative",
        confidence=float(prediction),
        prediction_id=str(uuid.uuid4())
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 8.4 A/B Testing Framework

**File:** `deployment/ab_testing.py`

```python
class ABTestingRouter:
    """Route traffic between model versions"""

    def __init__(self, model_a: str, model_b: str, split: float = 0.5):
        self.model_a = mlflow.pytorch.load_model(model_a)
        self.model_b = mlflow.pytorch.load_model(model_b)
        self.split = split

    def predict(self, features, user_id: str):
        """Route to model A or B based on user_id hash"""
        if hash(user_id) % 100 < self.split * 100:
            return self.model_a.predict(features), "model_a"
        else:
            return self.model_b.predict(features), "model_b"
```

**Deliverables:**
- [ ] Advanced embedding features
- [ ] Hyperparameter optimization with Optuna
- [ ] FastAPI model serving
- [ ] A/B testing framework
- [ ] Model performance comparison dashboard

---

## Phase 9: Testing & Documentation (Week 13)

### 9.1 Unit Tests

**File:** `tests/unit/test_feature_store.py`

```python
import pytest
from feature_store.feature_registry import FeatureRegistry, FeatureDefinition
from feature_store.storage_backends.delta_store import DeltaFeatureStore

def test_feature_registry():
    registry = FeatureRegistry()

    feature_def = FeatureDefinition(
        name="word_count",
        feature_type="numerical",
        description="Number of words",
        computation_fn=lambda x: len(x.split()),
        dependencies=[],
        version="1.0",
        owner="mlops",
        tags=["text"]
    )

    registry.register_feature(feature_def)
    assert registry.get_feature("word_count") == feature_def

def test_delta_store_write_read():
    store = DeltaFeatureStore()

    # Create test dataframe
    df = spark.createDataFrame([
        (1, "hello world", 2),
        (2, "test review", 2)
    ], ["id", "text", "word_count"])

    # Write and read
    store.write_features(df, "test_features")
    result = store.read_features("test_features")

    assert result.count() == 2
```

### 9.2 Integration Tests

**File:** `tests/integration/test_pipeline.py`

```python
def test_end_to_end_pipeline():
    """Test full pipeline from ingestion to prediction"""

    # 1. Ingest data
    # 2. Compute features
    # 3. Train model
    # 4. Make prediction
    pass
```

### 9.3 Documentation

**Generate documentation:**

```bash
# dbt docs
cd data_transformation && dbt docs generate && dbt docs serve

# API docs (FastAPI auto-generates)
# Available at http://localhost:8000/docs

# Sphinx for Python code
sphinx-quickstart
sphinx-apidoc -o docs/ .
make html
```

**Deliverables:**
- [ ] Comprehensive unit tests
- [ ] Integration tests for pipelines
- [ ] E2E testing framework
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] User guides and runbooks

---

## Implementation Roadmap Summary

### Quick Start (First 2 Weeks)

**Priority 1: Foundation**
1. Set up Docker infrastructure
2. Restructure project directories
3. Implement basic feature store with PostgreSQL
4. Create first Airflow DAG

**Priority 2: Feature Engineering**
5. Migrate existing feature extraction to modular design
6. Set up DVC for versioning
7. Integrate MLflow tracking

**Priority 3: Production Ready**
8. Add data validation
9. Implement dbt transformations
10. Set up monitoring

### Success Metrics

- **Pipeline Reliability:** 99%+ DAG success rate
- **Data Quality:** 100% of features pass validation
- **Model Performance:** Maintain >85% accuracy
- **Latency:** <100ms p95 prediction latency
- **Feature Freshness:** <24 hour lag

---

## Key Differences from Reference Architecture

### Adaptations for Sentiment Analysis:

1. **Simplified Streaming:** Start with batch, add streaming later
2. **Feature Store Focus:** Emphasize TF-IDF and embeddings over generic features
3. **Model Registry:** Central for sentiment models (sklearn → PyTorch transition)
4. **Monitoring:** Focus on accuracy drift vs. toxic comment specifics

### Technology Choices:

| Component | Reference Repo | Our Implementation | Reason |
|-----------|----------------|-------------------|--------|
| Storage | Delta Lake | Delta Lake + PostgreSQL | Hybrid approach |
| Processing | Spark + Flink | Start with Spark only | Incremental complexity |
| ML Framework | Not specified | PyTorch | Flexibility for NLP |
| Feature Store | Custom | Feast alternative | Lighter weight |

---

## Appendix: Key Configuration Files

### A. Environment Variables (`.env`)

```bash
# Database
DB_PASSWORD=your_secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# MinIO
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Airflow
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://mlops:${DB_PASSWORD}@postgres/airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor

# Spark
SPARK_MASTER=spark://spark-master:7077
```

### B. Python Requirements

```txt
# requirements.txt
pandas==2.0.0
scikit-learn==1.3.0
torch==2.0.0
transformers==4.30.0
nltk==3.8.1
spacy==3.5.0
pyspark==3.4.0
delta-spark==2.4.0
dbt-core==1.5.0
dbt-postgres==1.5.0
great-expectations==0.17.0
mlflow==2.8.0
dvc==3.0.0
apache-airflow==2.7.0
prometheus-client==0.17.0
fastapi==0.100.0
uvicorn==0.23.0
optuna==3.3.0
```

---

## Next Steps

1. **Review this plan** with your team
2. **Prioritize phases** based on immediate needs
3. **Set up development environment** (Docker Compose)
4. **Start with Phase 1** (Infrastructure)
5. **Iterate incrementally** - don't try to implement everything at once

**Recommended First Sprint (2 weeks):**
- Docker Compose setup
- Project restructuring
- Basic PostgreSQL feature store
- Migrate existing training script to MLflow

---

**Document Version:** 1.0
**Last Updated:** 2024
**Author:** MLOps Team
**Status:** Draft - Ready for Implementation
