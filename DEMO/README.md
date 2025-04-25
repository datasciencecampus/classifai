### Minimal Example for 'generic' classifAI_API microservice

---

##### What this does

* abstracts the core parts of the classifAI fastAPI utility to an extendable, importable microservice
* for this POC, it has an `/embed` endpoint which generates SOC embeddings for data passed to it (see `test_data.json` for an example)
* `example_extended_serverside.py` shows how a new server can be built to extend this with a new endpoint, `/new_endpoint_score`
* `example_clientside.py` shows how data would be passed to / recieved from this new, extended, server.
* Together, the demo serverside and clientside scripts replicate the existing functionality of the original classifAI repo's `/SOC` endpoint

---



##### Installation:

* the built python package is available in the `dist/` directory in the level above this demonstration directory
* (optional) set up temporary .demo_env virtual environment;
```
python -m venv .demo_env
source .demo_env/bin/activate
```
* install with `python -m pip install ./classifAI_API-0.0.1.tar.gz`
* later, use `twine` to publish this(or a future) package to Artifacts Registry;

`python3 -m twine upload --repository-url https://eu-west1-python.pkg.dev/PROJECT_ID/.../ minimal_classifai_api-0.0.1.tar.gz`

* when on Artifacts Registry, install with something like

`pip install --index-url https://eu-west1-python.pkg.dev/PROJECT_ID/.../ minimal_classifai_api`

##### Running the demo

1. get the necessary environmental variables from classifAI Google Secrets and update the .env file
2. run 
```
gcloud auth login
gcloud auth application-default login
gcloud config set project classifai-sandbox
```
3. (if on a Cloud Workstation) activate a python virtual environment (e.g. the one from the classifAI repo)
4. install the demo package as described above
5. use one terminal to run the server with `python example_extended_serverside.py`
6. once running, use another terminal to run the clientside script; `python example_clientside.py`
