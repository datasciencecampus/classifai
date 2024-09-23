# configure required providers
terraform {
  required_version = ">=1.9.4, <2"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.0.1"
    }
  }
}

# configure google provider defaults
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
