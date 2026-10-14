variable "subscription_id" {
  description = "Azure subscription to deploy into (az account show --query id -o tsv)."
  type        = string
}

variable "location" {
  description = <<-EOT
    Azure region for everything except, possibly, the model (see
    openai_location). Sweden Central keeps the EU AI Act corpus in the EU and
    carries the widest Azure OpenAI model catalogue of the EU regions.
  EOT
  type        = string
  default     = "swedencentral"
}

variable "name" {
  description = "Name prefix for every resource. Lowercase letters, digits and hyphens."
  type        = string
  default     = "evalgate-rag"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,16}$", var.name))
    error_message = "name must be 3-17 characters: lowercase letters, digits, hyphens, starting with a letter."
  }
}

variable "allowed_cidr" {
  description = <<-EOT
    CIDR permitted to reach the API. REQUIRED and deliberately without a
    default, for the same reason as the AWS stack: /query has no
    authentication, and every call spends model quota. Set this to
    "<your.ip>/32" (curl ifconfig.me).
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.allowed_cidr))
    error_message = "allowed_cidr must be valid CIDR notation, e.g. 203.0.113.4/32."
  }
}

variable "image" {
  description = <<-EOT
    Container image to run. Defaults to the image ci.yml publishes on every
    push to main. The GHCR package must be public: no registry credentials are
    configured. It must also be built from a commit that includes the
    `telemetry` extra, or the API fails at boot with TELEMETRY__ENABLED=true.
  EOT
  type        = string
  default     = "ghcr.io/binoy-sasmal/evalgate-rag:latest"
}

variable "openai_location" {
  description = <<-EOT
    Region for the Azure OpenAI resource. Null means var.location. Model
    availability and quota differ by region, so this is separate from where the
    rest of the stack runs.
  EOT
  type        = string
  default     = null
}

variable "openai_model" {
  description = <<-EOT
    Model to deploy, with its version and deployment SKU. A hosted model cannot
    be pinned the way a package can: Azure retires model versions on a
    published schedule, so this default WILL rot. Check what your region
    currently offers before applying:
      az cognitiveservices model list -l <region> -o table

    Choose a model that accepts `max_tokens` and `temperature`, which is what
    the client sends. Reasoning models reject both.
  EOT
  type = object({
    name    = string
    version = string
    sku     = string
  })
  default = {
    name    = "gpt-4o-mini"
    version = "2024-07-18"
    sku     = "GlobalStandard"
  }
}

variable "openai_capacity" {
  description = "Deployment rate limit in thousands of tokens per minute. This is a quota, not a charge."
  type        = number
  default     = 20
}

variable "min_replicas" {
  description = <<-EOT
    0 lets the API scale to zero, which keeps compute inside the Container Apps
    monthly free grant at demo traffic. The price is a cold start on the first
    request after idle, while the embedding model loads and the BM25 index is
    rebuilt. Set to 1 to keep one replica warm and pay for it around the clock.
  EOT
  type        = number
  default     = 0
}

variable "postgres_sku" {
  description = "PostgreSQL Flexible Server SKU. B_Standard_B1ms is the smallest burstable size."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "db_password_version" {
  description = <<-EOT
    Bump this to rotate the database password. The password is generated fresh
    on every run and never stored in state, so Terraform only sends it to
    PostgreSQL and Key Vault when this number changes.
  EOT
  type        = number
  default     = 1
}

variable "budget_amount" {
  description = "Monthly cost in the subscription's billing currency at which the budget alert fires."
  type        = number
  default     = 25
}

variable "budget_notification_email" {
  description = "Where the budget alert goes."
  type        = string
}
