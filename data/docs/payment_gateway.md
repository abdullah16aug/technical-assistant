# Microservice Architecture: Payment Gateway Integration
## Overview
The Payment Gateway Microservice handles all third-party transaction processing via Stripe and PayPal. It operates on port 8085.

## Error Handling & Codes
- **ERR_PAY_401**: Unauthorized access token. Occurs if the `X-Gateway-Token` is missing or expired.
- **ERR_PAY_504**: Gateway timeout. The internal timeout limit for connecting to Stripe API is set to 15000ms (15 seconds).

## Database Configuration
The microservice utilizes a Redis cache for transaction state management before committing to PostgreSQL. The cache expiration is configured via `REDIS_TTL_SECONDS`, defaulting to 300 seconds.

# Payment Gateway Integration Guide

## API Architecture
The payment gateway uses a microservices architecture. The core routing is handled by the API Gateway service, which defaults to a timeout of 30 seconds for downstream HTTP requests.

## Cache Infrastructure
To optimize transactional verification speeds, we deploy a specialized Redis cache container named `payment-redis-cache`. 
* **Default Port**: 6379
* **Eviction Policy**: allkeys-lru

If the cache container hangs or becomes unresponsive, downstream internal services will start throwing `504 Gateway Timeout` errors. To resolve this, engineers must execute a hard restart on the container using the Docker CLI command: `docker restart payment-redis-cache`.