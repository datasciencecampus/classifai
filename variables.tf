# project-level
variable "project_id" {
  description = "GCP project ID."
  type        = string
  nullable    = false
}
variable "account_id" {
  description = "GCP account ID."
  type        = string
  nullable    = false
}
variable "region" {
  description = "GCP region."
  type        = string
  nullable    = false
}
variable "zone" {
  description = "GCP zone."
  type        = string
  nullable    = false
}
variable "github_org_name" {
  description = "GitHub organisation name."
  type = string
  nullable = false
}
variable "github_repo_name" {
  description = "GitHub repository name."
  type = string
  nullable = false
}

# iap
variable "support_email" {
  description = "Support email."
  type        = string
  nullable    = false
}
variable "application_title" {
  description = "Application title."
  type        = string
  nullable    = false
}

# IAP configuration variables
variable "iap_conditions" {
  description = "Dict {member: title, description, condition}"
  type        = map(map(string))
  nullable    = false
}

variable "iap_open" {
  description = "List of principals with unconditional IAP pass"
  type = list(string)
  nullable = false
}

# iam
variable "iam_roles_owner" {
  description = "List of project owners"
  type = list(string)
  nullable = false
}

variable "iam_roles_additional" {
  description = "List of members for additional roles"
  type = list(string)
  nullable = false
}

variable "app_engine_service_account" {
  description = "App Engine Service Account"
  type = string
  nullable = false
}

variable "app_engine_service_account_roles" {
  description = "List of roles"
  type = list(string)
  nullable = false
}

variable "wip_service_account_roles" {
  description = "List of roles"
  type = list(string)
  nullable = false
}

variable "services" {
  type = list(string)
}

variable "compute_engine_service_account" {
  description = "Compute Engine Service Account"
  type = string
  nullable = false
}
variable "compute_engine_service_account_roles" {
  description = "List of roles"
  type = list(string)
  nullable = false
}
variable "app_engine_service_agent_roles" {
  description = "List of roles"
  type = list(string)
  nullable = false
}
variable "cloudbuild_legacy_service_agent_roles" {
  description = "List of roles"
  type = list(string)
  nullable = false
}
variable "pubsub_service_agent_roles" {
  description = "List of roles"
  type = list(string)
  nullable = false
}
