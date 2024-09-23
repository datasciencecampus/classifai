## First use
To use terraform on your MAC for the first time, you must carry out some simple first steps:

``` bash
$ brew tap hashicorp/tap
$ brew install hashicorp/tap/terraform

$ terraform init
```

## About the main.tf
The main terraform file is a script made up of individual recipes - one for each resource. The recipe comprises the command and parameters. Parameter values are referenced from the `variables.auto.tfvars` file with each variable described (metadata) in the `variables.tf` to ensure that values conform to the required format.

## Important learnings
- Adding to a resource which has previously been added to the GCP project manually may result in destructive behaviour i.e. previous values lost and replaced by new values rather than complementing each other.
- Service agents and service accounts are typically generated automatically when an API is activated. In this project, a destructive error occurred (as indicated above) requiring explicit management of each previous service account and service agent here.
- A `.tfvars` file can be pushed to the repo but all sensitive values or part-values (e.g. IP addresses) must be pulled from Secret Manager. N.B. As this is a private repo, this is less of a concern. However, it is noteworthy that the account_id is not considered sensitive and user e-mails are attributed to the repo when we push code anyway.

## Infrastucture as code
Infrastructure can be evaluated and updated as follows:
- `terraform plan` will evaluate differences between the local specs and the remote specs and construct a plan for updates. This gives as an opportunity for a developer to assess the plan.
- `terraform apply` will apply any necessary changes to bring the remote specs in line with local specs. The plan is generated as part of this command and can be reviewed before committing changes.

These configurations represent a template (or 'truth') which will be matched in the remote project. Unless changes have been made to the local configuration files, any developer running the above commands will receive the following message after evaluation:

<img title="" alt="" src="/terraform/images/no_changes.png">
