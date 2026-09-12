variable "region" {
  description = <<-EOT
    AWS region. Keep everything in one region; cross-AZ and cross-region
    transfer are billable even inside the Free Tier.

    eu-north-1 (Stockholm) note: it offers no t2 instances at all, so t3.micro
    is the Free Tier instance type here -- the 750 free hours cover "t2.micro,
    or t3.micro in regions where t2.micro is unavailable", which is this one.
    It is also among the cheaper regions once the Free Tier lapses, and serving
    an EU AI Act corpus from an EU region is a reasonable default on its own.
  EOT
  type        = string
  default     = "eu-north-1"
}

variable "name" {
  description = "Name prefix for every resource."
  type        = string
  default     = "evalgate-rag"
}

variable "instance_type" {
  description = <<-EOT
    Free Tier covers 750 hours/month of t2.micro or t3.micro (whichever the
    region offers) for the first 12 months of an account. Anything larger is
    billable from the first hour. 1GB RAM is the binding constraint here --
    see user_data.sh, which provisions swap because the API (ONNX ~350MB) and
    Postgres share this box.
  EOT
  type        = string
  default     = "t3.micro"
}

variable "allowed_cidr" {
  description = <<-EOT
    CIDR permitted to reach the API on port 80. REQUIRED and deliberately
    without a default.

    /query spends your Groq quota on every call, and the free tier is capped at
    1K requests and 100K tokens per day. An endpoint open to 0.0.0.0/0 has no
    authentication in front of it, so anyone who finds the IP can exhaust that
    quota -- which also takes the eval gate down with it, since the Ragas judge
    runs on the same key. Set this to "<your.ip>/32" (curl ifconfig.me).
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.allowed_cidr))
    error_message = "allowed_cidr must be valid CIDR notation, e.g. 203.0.113.4/32."
  }
}

variable "image" {
  description = <<-EOT
    Container image to run. Defaults to the image ci.yml already publishes on
    every push to main. The GHCR package must be public, or the instance cannot
    pull it -- there are no registry credentials on the box. Make it public once
    at: github.com/users/<you>/packages/container/evalgate-rag/settings
  EOT
  type        = string
  default     = "ghcr.io/binoy-sasmal/evalgate-rag:latest"
}

variable "llm_api_key_parameter" {
  description = <<-EOT
    Name of the SSM Parameter Store SecureString holding the Groq API key.

    Terraform reads nothing from it and never creates it -- a secret passed
    through a Terraform variable ends up in plaintext in the state file. Create
    it out of band (see docs/deploy-aws.md); this stack only grants the instance
    permission to read this one parameter. Standard parameters are free;
    Secrets Manager would be $0.40/secret/month.
  EOT
  type        = string
  default     = "/evalgate-rag/llm-api-key"
}

variable "llm_model" {
  description = "Model id passed to the OpenAI-compatible endpoint."
  type        = string
  default     = "qwen/qwen3.8-27b"
}

variable "llm_base_url" {
  description = "OpenAI-compatible base URL."
  type        = string
  default     = "https://api.groq.com/openai/v1"
}

variable "budget_notification_email" {
  description = <<-EOT
    Email for the $1 budget alarm. The whole point of this stack is a $0 bill,
    so the alarm is a tripwire: at $1 something is wrong (an instance outside
    the Free Tier, a second EIP, a forgotten snapshot), and you want to know
    within a day rather than at the end of the month.
  EOT
  type        = string
}

variable "root_volume_gb" {
  description = "Root EBS size. Free Tier covers 30GB of gp2/gp3 per month across all volumes."
  type        = number
  default     = 20
}

variable "ami_architecture" {
  description = <<-EOT
    x86_64 for t2/t3 (what the 12-month Free Tier covers), arm64 for t4g.
    Graviton's t4g.small free trial ended, so arm64 is a paid choice here --
    cheaper per hour, but not free.
  EOT
  type        = string
  default     = "x86_64"

  validation {
    condition     = contains(["x86_64", "arm64"], var.ami_architecture)
    error_message = "ami_architecture must be x86_64 or arm64."
  }
}

variable "ami_kernel" {
  description = <<-EOT
    AL2023 kernel line to select. "6.1" is the long-term default; newer images
    also publish as 6.12 and 6.18. Pinned because all variants are released the
    same day, so an unpinned "most recent" lookup can flip kernel lines between
    applies and replace the instance for no reason.
  EOT
  type        = string
  default     = "6.1"
}
