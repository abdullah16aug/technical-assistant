# GCP Data Engineering Knowledge Base

## Overview

This document contains internal best practices for the Data Engineering team working on Google Cloud Platform.

Technology Stack

- Google Cloud Platform
- BigQuery
- Cloud Storage (GCS)
- Pub/Sub
- Dataflow
- Dataproc
- Cloud Composer (Airflow)
- Databricks
- Apache Spark
- Python
- Terraform

---

# BigQuery Best Practices

## Optimize Query Cost

- Avoid `SELECT *`
- Query only required columns
- Use partitioned tables
- Cluster large tables
- Avoid unnecessary full table scans

Example

```sql
SELECT customer_id,total_amount
FROM sales
WHERE transaction_date >= '2026-01-01'
```

---

## Partitioning

Use partitioning for tables that continuously grow.

Recommended partition column

- transaction_date
- created_at
- ingestion_date

Benefits

- Lower cost
- Faster queries

---

## Clustering

Cluster by frequently filtered columns.

Examples

- customer_id
- country
- product_id

---

# Apache Spark Best Practices

## Executor Memory

Recommended

```
spark.executor.memory=8g
spark.driver.memory=4g
```

---

## Reduce Shuffle

Lower shuffle operations whenever possible.

Instead of

- groupByKey()

Prefer

- reduceByKey()
- aggregateByKey()

---

## Broadcast Join

Use broadcast joins when one table is small.

Example

```python
from pyspark.sql.functions import broadcast

result = orders.join(
    broadcast(customers),
    "customer_id"
)
```

---

## Cache Carefully

Cache only reused DataFrames.

Always unpersist after usage.

```python
df.cache()

...

df.unpersist()
```

---

# Airflow Best Practices

## Retry Policy

Recommended

- retries = 3
- retry_delay = 5 minutes

---

## DAG Naming

Use

```
customer_daily_pipeline
```

instead of

```
test1
```

---

## Logging

Always enable task logging.

Store logs in GCS.

---

# Cloud Composer

Common issues

- Scheduler stopped
- Missing dependencies
- Incorrect connections
- Worker out of memory

---

# Pub/Sub

Always acknowledge messages after successful processing.

Otherwise duplicate delivery occurs.

Pseudo flow

Receive Message

↓

Process

↓

Ack Message

---

# Dataflow

Enable autoscaling

Recommended worker type

```
n2-standard-4
```

Avoid expensive DoFn logic.

---

# Dataproc

Use autoscaling policies.

Terminate idle clusters.

Prefer ephemeral clusters for batch workloads.

---

# Cloud Storage

Recommended bucket layout

```
landing/

raw/

processed/

archive/
```

---

# IAM

Least privilege principle.

Common Roles

Storage Object Admin

BigQuery Data Editor

BigQuery Job User

Pub/Sub Publisher

Pub/Sub Subscriber

Composer Worker

---

# Monitoring

Use Cloud Monitoring dashboards.

Track

- Pipeline failures
- Worker CPU
- Memory
- DAG duration
- Failed jobs

---

# Error Handling

Always

- Retry transient failures
- Log stack traces
- Send alerts
- Fail gracefully

---

# Data Quality

Validate

- Null values
- Duplicate records
- Invalid dates
- Schema mismatch

---

# Performance Tips

BigQuery

✔ Partition tables

✔ Cluster tables

✔ Avoid SELECT *

Spark

✔ Broadcast joins

✔ Cache carefully

✔ Reduce shuffle

Airflow

✔ Small DAGs

✔ Modular tasks

✔ Retry failed jobs

---

# Security

Never store credentials inside source code.

Use

- Secret Manager
- Service Accounts
- IAM Roles

Rotate keys regularly.