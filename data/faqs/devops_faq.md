# DevOps & Infrastructure FAQ
## Q: How do I restart the staging environment cache?
A: SSH into the staging box and execute `docker restart payment-redis-cache`.

## Q: What should I do if I see an ERR_PAY_504 timeout error in production logs?
A: Check the Stripe Status dashboard first. If Stripe is operational, increase the `GATEWAY_TIMEOUT_MS` environment variable in the deployment chart to `20000` (20 seconds) and trigger a rolling restart.

# DevOps Infrastructure FAQ

### Q: Why am I getting a 504 Gateway Timeout error on staging?
A: A 504 error typically signifies that the API Gateway did not receive a timely response from the internal service layer. This is almost always caused by a hung `payment-redis-cache` instance failing to process session tokens.

### Q: How do I verify if Redis is down?
A: Run `docker ps` and check the status of `payment-redis-cache`. If the status is unhealthy or stuck, it needs to be restarted immediately.