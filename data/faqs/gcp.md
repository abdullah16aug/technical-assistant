# Frequently Asked Questions

## Q1

Question

How do I optimize BigQuery query performance?

Answer

Use partitioned tables, clustered tables, avoid SELECT *, and filter on partition columns.

---

## Q2

Question

Why is my Spark job failing with OutOfMemoryError?

Answer

Increase executor memory, reduce shuffle operations, and avoid caching unnecessary DataFrames.

---

## Q3

Question

How do I fix duplicate Pub/Sub messages?

Answer

Always acknowledge the message after successful processing.

---

## Q4

Question

How can I reduce BigQuery costs?

Answer

Avoid SELECT *, query only required columns, partition large tables, and cluster frequently filtered columns.

---

## Q5

Question

How do I optimize Spark joins?

Answer

Use Broadcast Join when one DataFrame is small to reduce shuffle.

---

## Q6

Question

Why is my Airflow DAG stuck in retry state?

Answer

Check task logs, verify external connections, ensure required services are running, and inspect scheduler health.

---

## Q7

Question

How do I fix Cloud SQL connection timeout from Airflow?

Answer

Verify the Cloud SQL Proxy is running and confirm firewall and IAM permissions.

---

## Q8

Question

What is the recommended Spark executor memory?

Answer

A common starting point is

spark.executor.memory=8g

Adjust based on workload.

---

## Q9

Question

How should I organize GCS buckets?

Answer

Use separate folders for

landing/

raw/

processed/

archive/

---

## Q10

Question

How do I improve Dataflow performance?

Answer

Enable autoscaling, choose appropriate worker types, and optimize expensive transforms.

---

## Q11

Question

Why is BigQuery slow?

Answer

Possible reasons

- Missing partitions
- Missing clustering
- Full table scans
- SELECT *

---

## Q12

Question

How do I fix Databricks OutOfMemoryError?

Answer

Increase cluster memory, reduce cache usage, call unpersist(), and optimize joins.

---

## Q13

Question

How do I secure GCP applications?

Answer

Use IAM roles, Secret Manager, and Service Accounts. Never hardcode credentials.

---

## Q14

Question

What are Spark performance best practices?

Answer

- Broadcast joins
- Reduce shuffle
- Avoid collect()
- Cache only reused DataFrames
- Use partition pruning

---

## Q15

Question

How do I monitor data pipelines?

Answer

Use Cloud Monitoring, Cloud Logging, Airflow logs, Datadog, or Prometheus dashboards.

---

## Q16

Question

What should I check if a Composer DAG fails?

Answer

- Scheduler health
- Worker logs
- Missing Python packages
- Connection configuration
- Task retries

---

## Q17

Question

How do I troubleshoot GCS permission errors?

Answer

Verify that the service account has the required IAM roles, such as Storage Object Creator or Storage Object Admin.

---

## Q18

Question

How do I optimize Spark DataFrame operations?

Answer

Filter early, select only required columns, avoid wide transformations when possible, and repartition only when necessary.

---

## Q19

Question

What is the best practice for handling schema changes in BigQuery?

Answer

Use schema evolution where appropriate, validate incoming data, and maintain versioned schemas for production pipelines.

---

## Q20

Question

How do I improve the reliability of data pipelines?

Answer

Implement retries, idempotent processing, monitoring, alerting, checkpointing, and comprehensive logging.