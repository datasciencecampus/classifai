---
title: Serving the API
date: 2024-08-08
status: decided
---

GCP architecture decisions are required to facilitate automation, data processing and serving the product to end-users.


## Use case/user story
The user can submit ASHE survey data to the product and SOC labels are returned.


## Problem/issue
Only certain users should have access to the service. The service should be cheap but scalable to bulk.


## Preferred option
App Engine - scalable from zero instances up; Cloud Build automated when app deployed; managed service; team experience with App Engine; greater control over user access


## Other options
1. Cloud Run - scalable from zero instance up; more laborious approach to limiting user access; team experience with Cloud Run; probably better suited to automation of process rather than as a product
2. Google Kubernetes Engine (GKE) - highly scalable; much greater technical expertise required to make more granular decisions on architecture; little team experience with GKE

## Desired consequences
The service is seamlessly updated when code changes are made.
The service is flexible and reproducible across numerous other use cases.

## Downside
Getting Google Embeddings (Google Generative AI API) to work in App Engine has been challenging given the need for an API key. What worked: running the `gunicorn` command with the service account flag `--service-account` followed by the app engine service account e-mail AND granting that service account with the Secret Manager Secret Accessor role AND replicating the API key as a Secret (GOOGLE_API_KEY). There are likely to be easier approaches.

Further issue - due to App Engine instances being distributed resources, a volume cannot be mounted as with Cloud Run or GKE. The vector store can be built in a `/tmp` folder but this is deleted as soon as an instance lapses. Consequently, the vector store would have to be rebuilt for every new instance. Alternatively, the vector store could be written separately to Cloud Storage and then ingested by an instance.

## Additional rationale
