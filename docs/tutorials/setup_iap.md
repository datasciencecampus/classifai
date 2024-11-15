## Setting up IAP for programmatic access between app versions
The following is based on the documentation [here](https://cloud.google.com/iap/docs/authentication-howto?_gl=1*9m1cgt*_ga*NTIxMTM5NTc5LjE3MjE5MDUzOTE.*_ga_WH2QY8WWF5*MTcyNTYyNTI1Ni44My4xLjE3MjU2MzEyOTguMzIuMC4w#authenticating_with_an_oidc_token).

### Create an IAP OAuth Client
In the **Credentials** section of **APIs and services** (in GCP Console), create a new credential (OAuth client ID) or we can select an existing one.

### Allowing programmatic access
1. We can 'permit' access for OAuth client IDs to GCP services - in this case, App Engine. For this, we require a **SETTING_FILE** in the project root directory. This can be generated from the Terminal:

```{bash}
cat << EOF > SETTING_FILE
access_settings:
  oauth_settings:
    programmatic_clients: ['<CLIENT_ID>']
EOF
```

**Note:** The client ID can easily be copied by accessing it in the GCP **Credentials** section referenced above. This is not the client secret! It's worth noting that multiple Client IDs can be entered [].

2. We allowlist this client ID on an App Engine service by executing a further command from the Terminal:

```{bash}
gcloud iap settings set SETTING_FILE --project=<PROJECT_ID> --resource-type=app-engine
```

### Authenticating for programmatic access using an OIDC token
In order to have one server (Flask) requesting a task in another server (FastAPI), a **headers** argument is defined in the corresponding `requests.request()` command. The value for this argument is acquired using the `_obtain_oidc_token()` method in our `flask_ui/app.py` script:

```{python}
def _obtain_oidc_token(oauth_client_id):
    """Obtain OIDC authentication token."""

    open_id_connect_token = id_token.fetch_id_token(Request(), oauth_client_id)
    headers = {"Authorization": "Bearer {}".format(open_id_connect_token)}

    return headers
```
