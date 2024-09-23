# project-level
project_id = "classifai-sandbox"
account_id = "14177695902"
region = "europe-west"
zone = "europe-west-2"

support_email = "edward.jackson@ons.gov.uk"
application_title = "ClassifAI"

# iap open
iap_open = [
    "edward.jackson@ons.gov.uk",
    "andrew.banks@ons.gov.uk",
    "mat.weldon@ons.gov.uk",
    "samuel.stock@ons.gov.uk",
    "li.chen@ons.gov.uk",
    "jyldyz.djumalieva@ons.gov.uk"
]
# iap conditions
iap_conditions = {
    "ethan.moss@ons.gov.uk" = {
        "title" = "Allow for a few hours",
        "description" = "Revoke access at 3pm on 2024-09-06",
        "expression" = "request.time < timestamp(\"2024-09-06T15:00:00Z\")"},
}

# iam roles
iam_roles_owner = [
      "user:jyldyz.djumalieva@ons.gov.uk",
      "user:edward.jackson@ons.gov.uk",
      "user:andrew.banks@ons.gov.uk",
      "user:lewis.edwards@ons.gov.uk",
      "user:mat.weldon@ons.gov.uk",
      "user:samuel.stock@ons.gov.uk",
      "user:li.chen@ons.gov.uk",
    ]

iam_roles_additional = [
    "user:edward.jackson@ons.gov.uk",
    ]

# service accounts and service agents
app_engine_service_account = "classifai-sandbox@appspot.gserviceaccount.com"

app_engine_service_account_roles = [
        "roles/appengine.appAdmin",
        "roles/cloudbuild.builds.builder",
        "roles/autoscaling.metricsWriter",
        "roles/compute.instanceAdmin.v1",
        "roles/firebase.managementServiceAgent",
        "roles/secretmanager.secretAccessor",
        "roles/iap.settingsAdmin",
        "roles/monitoring.editor",
        "roles/iam.serviceAccountUser"
    ]

compute_engine_service_account = "14177695902-compute@developer.gserviceaccount.com"
compute_engine_service_account_roles = [
        "roles/appengine.appAdmin",
        "roles/cloudbuild.builds.builder",
        "roles/autoscaling.metricsWriter",
        "roles/compute.instanceAdmin.v1",
        "roles/firebase.managementServiceAgent",
        "roles/secretmanager.secretAccessor",
        "roles/iap.settingsAdmin",
        "roles/monitoring.editor",
        "roles/iam.serviceAccountUser",
        "roles/iap.tunnelResourceAccessor"
        # "roles/editor"  # default has many excess permissions
    ]

services = [
    "aiplatform.googleapis.com",
    "compute.googleapis.com"
]

app_engine_service_agent_roles = [
    "roles/appengine.serviceAgent",
    "roles/secretmanager.secretAccessor",
]

cloudbuild_legacy_service_agent_roles = [
    "roles/cloudbuild.serviceAgent",
    "roles/storage.objectViewer",
]
pubsub_service_agent_roles = [
    "roles/pubsub.serviceAgent",
    "roles/iam.serviceAccountTokenCreator",
]
