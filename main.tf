# any refactoring of a resource may require it to be reset
# `terraform state list`
# `terraform state rm <RESOURCE.MEMBER>`


##################
# FIREWALL RULES #
##################

data "google_secret_manager_secret_version" "ej_ip" {
  provider = google-beta
  project = var.project_id
  secret  = "ej_ip"      # pragma: allowlist secret
  version = "latest"
}
data "google_secret_manager_secret_version" "ons_vpn_1" {
  provider = google-beta
  project = var.project_id
  secret  = "ons_vpn_1"      # pragma: allowlist secret
  version = "latest"
}
data "google_secret_manager_secret_version" "ons_vpn_2" {
  provider = google-beta
  project = var.project_id
  secret  = "ons_vpn_2"      # pragma: allowlist secret
  version = "latest"
}
data "google_secret_manager_secret_version" "ons_vpn_3" {
  provider = google-beta
  project = var.project_id
  secret  = "ons_vpn_3"      # pragma: allowlist secret
  version = "latest"
}

resource "google_app_engine_firewall_rule" "ons_vpn_1" {
  project      = var.project_id
  priority     = 1
  action       = "ALLOW"
  source_range = data.google_secret_manager_secret_version.ons_vpn_1.secret_data
  description = data.google_secret_manager_secret_version.ons_vpn_1.secret
}
resource "google_app_engine_firewall_rule" "ons_vpn_2" {
  project      = var.project_id
  priority     = 2
  action       = "ALLOW"
  source_range = data.google_secret_manager_secret_version.ons_vpn_2.secret_data
  description = data.google_secret_manager_secret_version.ons_vpn_2.secret
}
resource "google_app_engine_firewall_rule" "ons_vpn_3" {
  project      = var.project_id
  priority     = 3
  action       = "ALLOW"
  source_range = data.google_secret_manager_secret_version.ons_vpn_3.secret_data
  description = data.google_secret_manager_secret_version.ons_vpn_3.secret
}
resource "google_app_engine_firewall_rule" "ej_ip" {
  project      = var.project_id
  priority     = 4
  action       = "ALLOW"
  source_range = data.google_secret_manager_secret_version.ej_ip.secret_data
  description = data.google_secret_manager_secret_version.ej_ip.secret
}


##############################
# IDENTITY-AWARE PROXY (IAP) #
##############################

# activate IAP api
resource "google_project_service" "project_service_iap" {
  project = var.project_id
  service = "iap.googleapis.com"
}

# ALREADY EXISTS PRIOR TO TERRAFORM
# LEAVING ALONE FOR NOW!
# # create IAP brand (login screen)
# resource "google_iap_brand" "project_brand" {
#   support_email     = var.support_email
#   application_title = var.application_title
#   project           = var.project_id
# }

# IAP resources split for members with and without conditions
# condition forces replacement by terraform every time
resource "google_iap_web_type_app_engine_iam_member" "member_with_condition" {
  app_id = var.project_id
  role = "roles/iap.httpsResourceAccessor"
  for_each = var.iap_conditions
  member = "user:${each.key}"

  condition {
    title       = each.value["title"]
    description = each.value["description"]
    expression  = each.value["expression"]
  }
}

resource "google_iap_web_type_app_engine_iam_member" "member_no_condition" {
  app_id = var.project_id
  role = "roles/iap.httpsResourceAccessor"
  for_each = toset(var.iap_open)
  member = "user:${each.value}"
}


###############################
# IDENTITY & ACCESS MGT (IAM) #
###############################

data "google_iam_policy" "admin" {
  binding {
    role = "roles/owner"
    members = var.iam_roles_owner
  }
  binding {
    role = "roles/appengine.appAdmin"
    members = var.iam_roles_additional
  }
}

resource "google_project_iam_member" "app_engine_default_service_account" {
  project = var.project_id
  for_each = toset(var.app_engine_service_account_roles)
  role = each.value
  member = "serviceAccount:${var.app_engine_service_account}"
}

resource "google_project_iam_member" "compute_engine_default_service_account" {
  project = var.project_id
  for_each = toset(var.compute_engine_service_account_roles)
  role = each.value
  member = "serviceAccount:${var.compute_engine_service_account}"
}


#######################################################################
# SERVICE AGENTS ARE CREATED AUTOMATICALLY WHEN APIS ARE ACTIVATED ####
# THEY ARE GENERATED HERE ONLY BECAUSE OF A DESTRUCTIVE PRIOR COMMAND #
#######################################################################

# AI PLATFORM SERVICE AGENT
resource "google_project_iam_member" "aiplatform_primary" {
  project = var.project_id
  role    = "roles/aiplatform.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}

# CLOUDBUILD GCP SERVICE AGENT
resource "google_project_iam_member" "cloudbuild_primary" {
  project = var.project_id
  role    = "roles/cloudbuild.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-cloudbuild.iam.gserviceaccount.com"
}

# CLOUDBUILD LEGACY SERVICE AGENT
resource "google_project_iam_member" "cloudbuild_legacy_primary" {
  project = var.project_id
  for_each = toset(var.cloudbuild_legacy_service_agent_roles)
  role    = each.value
  member  = "serviceAccount:${var.account_id}@cloudbuild.gserviceaccount.com"
}

# APP ENGINE SERVICE AGENT
resource "google_project_iam_member" "appengine_primary" {
  project = var.project_id
  for_each = toset(var.app_engine_service_agent_roles)
  role    = each.value
  member  = "serviceAccount:service-${var.account_id}@gcp-gae-service.iam.gserviceaccount.com"
}

# COMPUTE ENGINE SERVICE AGENT
resource "google_project_iam_member" "computeengine_primary" {
  project = var.project_id
  role    = "roles/compute.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@compute-system.iam.gserviceaccount.com"
}

# CLOUD SERVICES SERVICE AGENT
resource "google_project_iam_member" "cloudservices_primary" {
  project = var.project_id
  role    = "roles/editor"
  member  = "serviceAccount:${var.account_id}@cloudservices.gserviceaccount.com"
}

# CLOUD AI PLATFORM NOTEBOOKS SERVICE AGENT
resource "google_project_iam_member" "notebooks_primary" {
  project = var.project_id
  role    = "roles/notebooks.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-notebooks.iam.gserviceaccount.com"
}

# CLOUD NETWORK MANAGEMENT SERVICE AGENT
resource "google_project_iam_member" "cloudnetmgt_primary" {
  project = var.project_id
  role    = "roles/networkmanagement.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-networkmanagement.iam.gserviceaccount.com"
}

# CLOUD WEB SECURITY SCANNER SERVICE AGENT
resource "google_project_iam_member" "cloudwebsecscan_primary" {
  project = var.project_id
  role    = "roles/websecurityscanner.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-websecurityscanner.iam.gserviceaccount.com"
}

# GOOGLE CONTAINER REGISTRY SERVICE AGENT
resource "google_project_iam_member" "containerregistry_primary" {
  project = var.project_id
  role    = "roles/containerregistry.ServiceAgent"
  member  = "serviceAccount:service-${var.account_id}@containerregistry.iam.gserviceaccount.com"
}

# ARTIFACT REGISTRY SERVICE AGENT
resource "google_project_iam_member" "artifactregistry_primary" {
  project = var.project_id
  role    = "roles/artifactregistry.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-artifactregistry.iam.gserviceaccount.com"
}

# CLOUD PUB/SUB SERVICE AGENT
resource "google_project_iam_member" "cloudpubsub_primary" {
  project = var.project_id
  for_each = toset(var.pubsub_service_agent_roles)
  role    = each.value
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# CLOUD FIRESTORE SERVICE AGENT
resource "google_project_iam_member" "firestore_primary" {
  project = var.project_id
  role    = "roles/firestore.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@gcp-sa-firestore.iam.gserviceaccount.com"
}

# GOOGLE CLOUD RUN SERVICE AGENT
resource "google_project_iam_member" "cloudrun_primary" {
  project = var.project_id
  role    = "roles/run.serviceAgent"
  member  = "serviceAccount:service-${var.account_id}@serverless-robot-prod.iam.gserviceaccount.com"
}
