# Architecture Comparison: Current vs. Reference System

## Executive Summary

This document provides a high-level comparison between the current **Sentiment-Analysis** codebase and the production-grade architecture from **Data-Systems-for-Toxic-Comment-Classification** repository.

---

## Current Architecture (Sentiment-Analysis)

### Overview
A basic, linear ML pipeline for sentiment analysis using scikit-learn.

### Architecture Diagram

```
┌─────────────────┐
│  Raw CSV Data   │
│ (IMDB Dataset)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  eda.ipynb      │
│  - HTML cleaning│
│  - Lemmatization│
│  - Stopwords    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessed CSV│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   train.py      │
│  - TF-IDF       │
│  - Train/Test   │
│  - 2 Models     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Joblib Models   │
│ (local files)   │
└─────────────────┘
```

### Components

| Component | Implementation | Technology |
|-----------|----------------|------------|
| **Data Storage** | Local CSV files | Filesystem |
| **Data Processing** | Single-node Python | Pandas |
| **Feature Engineering** | In-memory TF-IDF | scikit-learn |
| **Model Training** | Sequential script | scikit-learn |
| **Model Storage** | Local .joblib files | Joblib |
| **Orchestration** | Manual execution | None |
| **Monitoring** | None | None |
| **Versioning** | Git only | Git |
| **Testing** | None | None |

### Strengths
✅ Simple and easy to understand
✅ Quick to prototype
✅ Minimal dependencies
✅ Good for learning and experimentation

### Limitations
❌ Not production-ready
❌ No scalability
❌ No feature versioning
❌ No data quality checks
❌ No model monitoring
❌ No orchestration
❌ Hardcoded file paths
❌ Single machine bottleneck

---

## Reference Architecture (Toxic Comment Classification)

### Overview
Production-grade ML system with feature store, orchestration, and comprehensive monitoring.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                               │
│                    (CSV, PostgreSQL, APIs)                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────┐                        ┌───────────────┐
│ BATCH         │                        │ STREAM        │
│ PROCESSING    │                        │ PROCESSING    │
│               │                        │               │
│ ┌───────────┐ │                        │ ┌───────────┐ │
│ │  Spark    │ │                        │ │  Debezium │ │
│ │  Jobs     │ │                        │ │   (CDC)   │ │
│ └─────┬─────┘ │                        │ └─────┬─────┘ │
│       │       │                        │       │       │
│       ▼       │                        │       ▼       │
│ ┌───────────┐ │                        │ ┌───────────┐ │
│ │Great      │ │                        │ │  Kafka    │ │
│ │Expect.    │ │                        │ │  Topics   │ │
│ └─────┬─────┘ │                        │ └─────┬─────┘ │
└───────┼───────┘                        │       │       │
        │                                │       ▼       │
        │                                │ ┌───────────┐ │
        │                                │ │  Flink    │ │
        │                                │ │  Jobs     │ │
        │                                │ └─────┬─────┘ │
        │                                └───────┼───────┘
        │                                        │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │     FEATURE STORE              │
        │                                │
        │  ┌──────────┐   ┌───────────┐ │
        │  │Delta Lake│   │PostgreSQL │ │
        │  │ (MinIO)  │   │  (Serving)│ │
        │  └──────────┘   └───────────┘ │
        │                                │
        │  ┌─────────────────────────┐  │
        │  │  Feature Registry       │  │
        │  │  - Metadata             │  │
        │  │  - Versioning           │  │
        │  │  - Lineage              │  │
        │  └─────────────────────────┘  │
        └────────────┬───────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│   dbt         │         │  TRINO        │
│ TRANSFORMATION│         │ (Federated    │
│               │         │  Queries)     │
│ ┌───────────┐ │         └───────┬───────┘
│ │ Staging   │ │                 │
│ │ Models    │ │                 │
│ └─────┬─────┘ │                 │
│       │       │                 │
│       ▼       │                 │
│ ┌───────────┐ │                 │
│ │Production │ │                 │
│ │ Marts     │ │                 │
│ └───────────┘ │                 │
└───────┬───────┘                 │
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   ML PIPELINE          │
        │                        │
        │  ┌──────────────────┐  │
        │  │  DVC Pipeline    │  │
        │  │  - Extract       │  │
        │  │  - Feature Eng   │  │
        │  │  - Train         │  │
        │  │  - Evaluate      │  │
        │  └────────┬─────────┘  │
        │           │            │
        │           ▼            │
        │  ┌──────────────────┐  │
        │  │  MLflow          │  │
        │  │  - Tracking      │  │
        │  │  - Registry      │  │
        │  │  - Serving       │  │
        │  └──────────────────┘  │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   ORCHESTRATION        │
        │                        │
        │  ┌──────────────────┐  │
        │  │ Apache Airflow   │  │
        │  │  - DAGs          │  │
        │  │  - Scheduling    │  │
        │  │  - Monitoring    │  │
        │  └──────────────────┘  │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   MONITORING           │
        │                        │
        │  ┌─────────┐           │
        │  │Prometheus◄───────┐  │
        │  └────┬────┘         │  │
        │       │              │  │
        │       ▼              │  │
        │  ┌─────────┐         │  │
        │  │ Grafana │         │  │
        │  └─────────┘         │  │
        │                      │  │
        │  ┌──────────────┐   │  │
        │  │ ELK Stack    │   │  │
        │  │ (Logs)       │   │  │
        │  └──────────────┘   │  │
        │                      │  │
        │  ┌──────────────┐   │  │
        │  │ Alertmanager ◄───┘  │
        │  └──────────────┘       │
        └─────────────────────────┘
```

### Components

| Component | Implementation | Technology |
|-----------|----------------|------------|
| **Data Storage** | Distributed, versioned | Delta Lake + PostgreSQL + MinIO |
| **Data Processing** | Distributed batch + streaming | PySpark + Apache Flink |
| **Feature Engineering** | Feature store with registry | Custom + Delta Lake |
| **Data Transformation** | SQL-based transformations | dbt |
| **Data Quality** | Automated validation | Great Expectations |
| **Model Training** | Versioned, tracked | DVC + MLflow |
| **Model Registry** | Centralized, staged | MLflow Registry |
| **Orchestration** | Workflow automation | Apache Airflow |
| **Monitoring** | Metrics, logs, alerts | Prometheus + Grafana + ELK |
| **Query Layer** | Federated queries | Trino |
| **Deployment** | Containerized | Docker Compose |

### Strengths
✅ Production-ready
✅ Scalable architecture
✅ Feature versioning and lineage
✅ Data quality enforcement
✅ Comprehensive monitoring
✅ Automated orchestration
✅ Model lifecycle management
✅ Real-time and batch processing
✅ Infrastructure as code
✅ Observability at every layer

### Complexity Trade-offs
⚠️ Steep learning curve
⚠️ Higher infrastructure costs
⚠️ More moving parts
⚠️ Requires DevOps expertise

---

## Key Architectural Differences

### 1. Data Storage Layer

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Format** | CSV files | Delta Lake (Parquet) | 100x faster queries |
| **Location** | Local disk | Distributed object storage (MinIO) | Scalability |
| **Versioning** | None | Time-travel, ACID transactions | Reproducibility |
| **Access Pattern** | Full file reads | Columnar, predicate pushdown | Performance |

**Migration Path:** Start with PostgreSQL, add Delta Lake later for historical features.

---

### 2. Feature Engineering

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Computation** | In-memory (pandas) | Distributed (Spark) | Handle 1000x data |
| **Storage** | Transient | Persistent feature store | Reusability |
| **Versioning** | None | Tracked with DVC/Delta | Reproducibility |
| **Serving** | Recompute each time | Pre-computed, indexed | <10ms retrieval |
| **Documentation** | Code comments | Feature registry with metadata | Discoverability |

**Migration Path:**
1. Extract feature computation into separate functions
2. Store features in PostgreSQL tables
3. Build feature registry
4. Add Spark for large-scale computation

---

### 3. Model Lifecycle

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Tracking** | None | MLflow experiments | Compare 100+ runs |
| **Versioning** | Git + joblib files | DVC + MLflow Registry | Proper lineage |
| **Deployment** | Manual copy | Staged transitions (Staging → Production) | Safety |
| **Rollback** | Manual | One-click rollback | Reduced downtime |
| **A/B Testing** | Not possible | Built-in routing | Data-driven decisions |

**Migration Path:**
1. Add MLflow tracking to train.py
2. Log all hyperparameters and metrics
3. Implement model registry
4. Create deployment automation

---

### 4. Orchestration & Automation

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Scheduling** | Manual | Airflow DAGs | Automated retraining |
| **Dependency Mgmt** | None | Task dependencies in DAGs | Correct execution order |
| **Failure Handling** | Script crashes | Retries, alerts, monitoring | Reliability |
| **Monitoring** | None | DAG-level metrics | Visibility |

**Migration Path:**
1. Create simple Airflow DAG for daily data processing
2. Add model training DAG with dependencies
3. Implement monitoring and alerting

---

### 5. Data Quality

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Validation** | None | Great Expectations | Prevent bad data |
| **Schema Checks** | Runtime errors | Explicit contracts | Early detection |
| **Statistical Tests** | None | Drift detection | Model degradation alerts |
| **Documentation** | None | Auto-generated data docs | Understanding |

**Migration Path:**
1. Define expectation suites for raw data
2. Add validation checkpoints in pipeline
3. Integrate with Airflow for automated checks

---

### 6. Monitoring & Observability

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Model Metrics** | Training time only | Continuous tracking | Detect degradation |
| **Data Drift** | None | Statistical monitoring | Retraining triggers |
| **Infrastructure** | None | Prometheus + Grafana | Resource optimization |
| **Alerting** | None | Alertmanager | Proactive response |
| **Logging** | Print statements | Structured logs (ELK) | Debugging |

**Migration Path:**
1. Add Prometheus metrics exporter
2. Create Grafana dashboards
3. Define alert rules
4. Integrate with Airflow

---

### 7. Deployment & Infrastructure

| Aspect | Current | Reference | Impact |
|--------|---------|-----------|--------|
| **Environment** | Local Python | Docker Compose | Reproducibility |
| **Dependencies** | requirements.txt | Container images | Isolation |
| **Scalability** | Single machine | Distributed services | Handle growth |
| **Configuration** | Hardcoded | Environment variables | Flexibility |

**Migration Path:**
1. Create Docker Compose for core services
2. Containerize application components
3. Use .env for configuration
4. Document deployment process

---

## Feature Store Deep Dive

### Why a Feature Store?

**Problems Solved:**
1. **Feature Reuse:** Compute once, use everywhere (training, serving, experiments)
2. **Consistency:** Same feature computation in training and production
3. **Discovery:** Catalog of available features with documentation
4. **Versioning:** Track feature changes over time
5. **Monitoring:** Detect feature drift and data quality issues
6. **Performance:** Pre-computed features for low-latency serving

### Reference Implementation Analysis

The toxic comment repo implements a **hybrid feature store**:

**Storage Tiers:**
```
Raw Data → Bronze (Delta Lake) → Silver (Delta Lake) → Gold (PostgreSQL)
           ↓                      ↓                      ↓
         Historical            Processed             Serving
```

**Key Components:**

1. **Delta Lake (Bronze/Silver):**
   - Stores historical feature snapshots
   - Enables time-travel queries
   - ACID transactions for consistency
   - Efficient columnar format

2. **PostgreSQL (Gold):**
   - Production feature serving
   - Low-latency point queries
   - Indexed for fast lookups
   - Supports real-time updates

3. **Feature Registry:**
   - Metadata about each feature
   - Computation logic
   - Dependencies
   - Ownership and documentation

4. **dbt Transformations:**
   - SQL-based feature engineering
   - Staging → Production promotion
   - Built-in testing
   - Documentation generation

### Implementation Recommendation

For Sentiment-Analysis, implement a **simplified feature store** first:

**Phase 1: Basic Feature Store (Week 3-4)**
```python
# PostgreSQL-based feature store
features/
  - text_statistics/     (word_count, char_count, etc.)
  - tfidf_features/      (TF-IDF vectors)
  - sentiment_scores/    (polarity, subjectivity)

# Schema
CREATE TABLE features.text_statistics (
    review_id VARCHAR PRIMARY KEY,
    word_count INT,
    char_count INT,
    avg_word_length FLOAT,
    feature_timestamp TIMESTAMP,
    feature_version VARCHAR
);
```

**Phase 2: Add Delta Lake (Week 5-6)**
- Store historical features in Delta format
- Enable time-travel for experiments
- Better performance for analytical queries

**Phase 3: Feature Registry (Week 7-8)**
- Catalog features with metadata
- Track feature lineage
- Document computation logic

---

## ML Pipeline Comparison

### Current Pipeline

```python
# train.py (linear execution)
1. Load CSV
2. Train/test split
3. Fit TfidfVectorizer
4. Train Decision Tree
5. Train Random Forest
6. Save models with joblib
```

**Limitations:**
- No experiment tracking
- No hyperparameter optimization
- Single evaluation metric
- No model comparison
- Manual model management

### Reference Pipeline

```yaml
# DVC pipeline (dvc.yaml)
stages:
  extract_data:
    deps: [feature_store]
    outs: [training_data.csv]

  feature_engineering:
    deps: [training_data.csv]
    outs: [features.parquet]

  train_model:
    deps: [features.parquet]
    params: [hyperparameters]
    metrics: [train_metrics.json]
    outs: [model.pth]

  evaluate_model:
    deps: [model.pth]
    metrics: [eval_metrics.json]
    plots: [confusion_matrix.png, roc_curve.png]
```

**With MLflow Tracking:**
```python
with mlflow.start_run():
    mlflow.log_params(hyperparameters)
    mlflow.log_metrics({"accuracy": 0.92})
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifacts("plots/")
```

**Advantages:**
- Reproducible pipeline execution
- Automatic caching (skip unchanged stages)
- Experiment comparison
- Metric visualization
- Model versioning and registry
- Artifact tracking

### Migration Strategy

**Week 1-2: Add MLflow Tracking**
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment-analysis")

with mlflow.start_run():
    # Existing training code
    mlflow.log_params({"model": "random_forest"})
    mlflow.log_metrics({"accuracy": accuracy})
    mlflow.sklearn.log_model(model, "model")
```

**Week 3-4: Implement DVC Pipeline**
```bash
dvc init
dvc run -n train -d train.py -o weights/model.joblib python train.py
dvc repro
```

**Week 5-6: Add Model Registry**
```python
mlflow.register_model("runs:/abc123/model", "sentiment_classifier")
mlflow.transition_model_version_stage("sentiment_classifier", 1, "Production")
```

---

## Technology Stack Recommendations

### Immediate Adoption (Week 1-4)

| Technology | Purpose | Priority | Complexity |
|------------|---------|----------|------------|
| **Docker Compose** | Local development environment | HIGH | Low |
| **PostgreSQL** | Feature store database | HIGH | Low |
| **MLflow** | Experiment tracking | HIGH | Low |
| **DVC** | Data/model versioning | HIGH | Medium |

### Near-term (Week 5-8)

| Technology | Purpose | Priority | Complexity |
|------------|---------|----------|------------|
| **Apache Airflow** | Workflow orchestration | HIGH | Medium |
| **dbt** | Data transformations | MEDIUM | Low |
| **Great Expectations** | Data validation | MEDIUM | Medium |
| **Prometheus + Grafana** | Monitoring | MEDIUM | Medium |

### Long-term (Week 9+)

| Technology | Purpose | Priority | Complexity |
|------------|---------|----------|------------|
| **Delta Lake** | Historical feature storage | MEDIUM | Medium |
| **MinIO** | Object storage | MEDIUM | Low |
| **Apache Spark** | Distributed processing | LOW | High |
| **Kafka + Flink** | Stream processing | LOW | High |
| **Trino** | Federated queries | LOW | Medium |

---

## Migration Risks & Mitigation

### Risk 1: Over-Engineering Too Early
**Risk:** Implementing complex infrastructure before it's needed
**Mitigation:**
- Start with PostgreSQL feature store, not Delta Lake
- Use Airflow LocalExecutor before Kubernetes
- Implement batch processing before streaming

### Risk 2: Learning Curve Overload
**Risk:** Team overwhelmed by new technologies
**Mitigation:**
- Introduce one tool at a time
- Provide hands-on training
- Start with toy examples before production data

### Risk 3: Infrastructure Costs
**Risk:** Running expensive distributed systems
**Mitigation:**
- Use Docker Compose for local development
- Cloud deployment only when scaling needed
- Monitor resource usage with Prometheus

### Risk 4: Breaking Existing Workflows
**Risk:** Migration disrupts current research
**Mitigation:**
- Keep existing train.py working
- Build new system in parallel
- Gradual cutover with feature flags

---

## Success Metrics

### Phase 1: Foundation (Weeks 1-4)
- [ ] Docker Compose infrastructure running
- [ ] All services accessible (PostgreSQL, MLflow, MinIO)
- [ ] Feature store schema defined
- [ ] First MLflow experiment logged

### Phase 2: Feature Engineering (Weeks 5-8)
- [ ] 10+ features in feature store
- [ ] DVC pipeline operational
- [ ] dbt models transforming data
- [ ] Great Expectations validating data

### Phase 3: Production Ready (Weeks 9-13)
- [ ] Airflow DAGs scheduling pipelines
- [ ] Model registry with 3+ versions
- [ ] Monitoring dashboards live
- [ ] <100ms feature retrieval latency

### Long-term Goals (3-6 months)
- [ ] 99% pipeline reliability
- [ ] Automated retraining on drift detection
- [ ] A/B testing framework operational
- [ ] Real-time inference API (<50ms p95)

---

## Key Takeaways

### What to Adopt Immediately
1. **Feature Store Pattern** - Even with PostgreSQL, this is valuable
2. **MLflow Tracking** - Minimal overhead, huge value
3. **DVC for Versioning** - Essential for reproducibility
4. **Docker Compose** - Simplifies environment setup

### What to Delay
1. **Delta Lake** - PostgreSQL is sufficient initially
2. **Stream Processing** - Batch is enough for most use cases
3. **Distributed Spark** - Single-node processing works at moderate scale
4. **Trino** - Not needed until multiple data sources

### Architecture Principles to Follow
1. **Start Simple, Scale When Needed** - Don't over-engineer
2. **Automate Everything** - Manual processes don't scale
3. **Monitor from Day One** - You can't fix what you can't see
4. **Version All Artifacts** - Data, features, models, code
5. **Separate Concerns** - Feature engineering ≠ Model training ≠ Serving

---

## Quick Start Guide

### Day 1: Setup Infrastructure
```bash
# Clone repo and setup
git clone https://github.com/your-org/Sentiment-Analysis
cd Sentiment-Analysis

# Create environment
cp .env.example .env
# Edit .env with your values

# Start infrastructure
docker-compose up -d

# Verify services
./scripts/check_services.sh
```

### Day 2: Setup Feature Store
```bash
# Create database schema
python scripts/setup_database.py

# Migrate existing data
python scripts/migrate_existing_data.py

# Test feature store
python tests/integration/test_feature_store.py
```

### Day 3: Setup MLflow
```bash
# Configure MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run first tracked experiment
python model_experiment/train.py

# View in browser
open http://localhost:5000
```

### Week 1 Goal
- Infrastructure running
- Feature store operational
- First MLflow experiment logged
- Team trained on basic concepts

---

## Conclusion

The reference architecture provides a **comprehensive blueprint** for building a production-grade ML system. The key is to **adopt incrementally**, starting with high-value, low-complexity components and progressively adding sophistication as needs grow.

**Recommended 3-Month Roadmap:**

**Month 1: Foundation**
- Docker infrastructure
- PostgreSQL feature store
- MLflow tracking
- Basic Airflow DAGs

**Month 2: Enhancement**
- dbt transformations
- Data validation
- Monitoring dashboards
- DVC pipeline

**Month 3: Production**
- Model registry
- Automated deployments
- Performance optimization
- Documentation

By following this path, you'll build a **scalable, maintainable, production-ready ML system** without overwhelming your team or over-engineering prematurely.

---

**Next Steps:**
1. Review this comparison with your team
2. Prioritize features based on your specific needs
3. Set up a pilot project with Docker + MLflow + PostgreSQL
4. Iterate based on feedback and lessons learned

**Questions to Discuss:**
- What's our target scale (data volume, request rate)?
- Do we need real-time predictions or batch inference?
- What's our tolerance for infrastructure complexity?
- What's our team's expertise with these technologies?
- What's our budget for infrastructure costs?
