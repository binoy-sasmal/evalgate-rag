# Amazon Linux 2023, resolved by image name so the AMI id is never hardcoded --
# ids are region-specific and rotate on every patch release.
#
# The usual idiom is the /aws/service/ami-amazon-linux-latest/... SSM alias,
# which works and is three lines instead of twenty. It is not used here for one
# reason: that alias tracks whatever AWS currently calls "default", which today
# means kernel 6.18. AL2023 publishes 6.1, 6.12 and 6.18 on the same date, so
# when the default advances, the alias silently returns a different image and
# the next apply replaces the instance -- a kernel major-version jump arriving
# as an unrelated diff. Pinning the kernel line makes that a deliberate edit to
# var.ami_kernel instead.
#
# The tradeoff, stated plainly: if AWS ever stops publishing the pinned line,
# this data source finds nothing and fails at plan time. That is the better
# failure -- it is loud, and it happens before anything is replaced.
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  # The kernel line is pinned rather than taking whatever is newest. All three
  # kernel variants publish on the same day, so "most recent" alone can flip
  # between them between applies and silently replace the instance. 6.1 is
  # AL2023's long-term default; bump var.ami_kernel deliberately.
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-kernel-${var.ami_kernel}-${var.ami_architecture}"]
  }

  filter {
    name   = "architecture"
    values = [var.ami_architecture]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "aws_instance" "this" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  # Require IMDSv2. The instance profile can read the LLM key parameter, and
  # IMDSv1's unauthenticated GET is the standard path from an SSRF in a public
  # web service to those credentials.
  metadata_options {
    http_tokens   = "required"
    http_endpoint = "enabled"
    # 1 hop: only the host itself talks to AWS (the SSM key fetch at boot).
    # Containers reach IMDS over the docker bridge, which costs a second hop --
    # so this also stops a compromised API container from assuming the
    # instance role and reading the LLM key parameter directly.
    http_put_response_hop_limit = 1
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    image                 = var.image
    region                = var.region
    llm_api_key_parameter = var.llm_api_key_parameter
    llm_base_url          = var.llm_base_url
    llm_model             = var.llm_model
  })

  # The instance is cattle: all state is either in the image (corpus, model) or
  # reproducible by re-running ingest, which user_data does on every boot. So a
  # config change should rebuild the box rather than leave it drifted from the
  # code that describes it. Re-ingest costs ~2 minutes of CPU and zero dollars.
  user_data_replace_on_change = true

  tags = { Name = var.name }
}

# A stable address that survives stop/start.
#
# Not free: since February 2024 AWS bills $0.005/hour for every public IPv4
# address -- Elastic or auto-assigned, instance running or stopped -- so this
# is ~$3.65/month and dropping the EIP for an auto-assigned address would save
# nothing. It is also why stopping the instance does not stop the meter: the
# address and the EBS volume keep billing. Use terraform destroy instead.
resource "aws_eip" "this" {
  instance = aws_instance.this.id
  domain   = "vpc"

  tags       = { Name = var.name }
  depends_on = [aws_internet_gateway.this]
}
