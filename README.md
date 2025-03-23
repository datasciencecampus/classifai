<img src="https://github.com/datasciencecampus/awesome-campus/blob/master/ons_dsc_logo.png"/>

# ClassifAI

A web API and user interface for coding free-text survey responses to statistical classifications (SIC, SOC, etc.)

This codebase includes:

* A FastAPI app that accepts free-text records, sends them to a third-party service for embedding, and then queries against a vector database to find top-N closest results
* A web user interface Flask app
* A Python package of core functions
* Cloud configuration files for App Engine, Cloud Build and GitHub Actions
* Scripts for evaluation

Terraform IaC is stored in a different repo.

## Installation

To clone the repo:

``` bash
git clone https://github.com/datasciencecampus/classifai.git
cd classifai
```

To install:

```bash
python -m venv venv
source venv/bin/activate # source venv/Scripts/activate on Windows
python -m pip install --upgrade pip
python -m pip install .
```

For development, install as editable with dev dependencies, and install pre-commit hooks:

``` bash
python -m pip install -e ".[dev]"
python -m pip install pre-commit
pre-commit install
```

## Authentication

To call Google embeddings a Google embedding API key is needed.

* To run the API locally the key can be added to `.env`
* In App Engine the server looks for a Google secret named `GOOGLE_API_KEY`

## Running the API and front end locally

Authenticate using gcloud:

```bash
gcloud config set project <GCP-project-ID>
gcloud auth login
gcloud auth application-default login
```

Set the following environment variables either in a `.env` file or directly:

```bash
ENV_TYPE=local
API_TYPE=local
PROJECT_ID=<GCP-project-ID>
# The following only needed for API
BUCKET_NAME=<name of bucket containing vector database>
EMBEDDING_API_KEY=<Google API key for embedding>
# The following only needed for front end
OAUTH_CLIENT_ID=<Client ID to authenticate local front end to API>
API_URL=<URL of API>
```

`API_URL` defaults to `http://localhost:8000` when `API_TYPE=local` so can be left unset.

In one terminal, run the API:

```bash
uvicorn fast_api.main:app
```

In another terminal, run the front end:

```bash
python -m flask --app flask_ui/app.py run
```

### Deploying the API and app

Both apps deploy automatically on merge for the 'dev', 'uat' and 'prod' branches in GitHub. This is governed by the three config files: `app.cloudbuild.[dev|uat|prod].yaml`

To deploy a single app manually to a temporary URL, use:

```bash
gcloud app deploy --no-promote app.[flask|fastapi].[dev|uat|prod].yaml
```

### Cloud set-up

This is not a complete set-up guide, just a list of a few gotchas:

* To work online the vdb must be deployed to a bucket, and the bucket name included in a secret called `APP_DATA_BUCKET`
* There are currently name mismatches between environment variables and the corresponding Google Secrets:

   - BUCKET_NAME --> APP_DATA_BUCKET
   - EMBEDDING_API_KEY --> GOOGLE_API_KEY
   - OAUTH_CLIENT_ID --> app_oauth_client_id

* To deploy successfully to App Engine the 'requirements.txt' must be up to date - use pip compile. App Engine cannot use the pyproject.toml

### REST API

REST APIs are deployed using GCP App Engine. Development is in its early stages. However, 3 GCP-based (but restricted) deployments of the app are available:

[**PROD**](https://ons-dsc-classifai-prod.nw.r.appspot.com)

[**UAT**](https://preview-dot-ons-dsc-classifai-prod.nw.r.appspot.com)


[**DEV**](https://dev-flask-dot-classifai-sandbox.nw.r.appspot.com)





### Utilities for developing the vector knowledgebase

A separate utility class is available to copy the ChromaDB vector store files/cache to a GCS bucket. In future, APP Engine instances will draw from this bucket to avoid having to recreate the vector store (the SOC source input is unlikely to change very often). To copy the current ChromaDB files, the following code can be used programmatically:

``` python
from src.classifai.utils import DB_Updater

tool = DB_Updater()
tool.update()
```

A utility is available to trigger a job to create a ChromaDB collection for either the **sic** or **soc** task. The collection is created in a tmp folder (`tmp/db/`) locally and automatically pushed to a GCS bucket.

``` python
from src.classifai.utils import setup_vector_store
setup_vector_store("sic") # or "soc"
```
# Data Science Campus

At the [Data Science Campus](https://datasciencecampus.ons.gov.uk/about-us/) we apply data science, and build skills, for public good across the UK and internationally. Get in touch with the Campus at [datasciencecampus\@ons.gov.uk](datasciencecampus@ons.gov.uk).

# License

<!-- Unless stated otherwise, the codebase is released under [the MIT Licence][mit]. -->

The code, unless otherwise stated, is released under [the MIT Licence](LICENCE).

The documentation for this work is subject to [© Crown copyright](http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/) and is available under the terms of the [Open Government 3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) licence.
