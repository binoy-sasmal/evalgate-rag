# Instance role: SSM Session Manager for shell access, plus read access to
# exactly one parameter -- the LLM API key.

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.name}-instance"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# Grants Session Manager (and only that); no inbound port, no key pair.
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "read_llm_key" {
  statement {
    actions = ["ssm:GetParameter"]
    # Scoped to the single parameter, not ssm:* or a wildcard path: this role
    # is attached to an instance exposed to the internet, so its blast radius
    # on compromise should be one key, not the whole parameter store.
    resources = [
      "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${var.llm_api_key_parameter}"
    ]
  }

  statement {
    actions   = ["kms:Decrypt"]
    resources = ["arn:aws:kms:${var.region}:${data.aws_caller_identity.current.account_id}:alias/aws/ssm"]
  }
}

resource "aws_iam_role_policy" "read_llm_key" {
  name   = "${var.name}-read-llm-key"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.read_llm_key.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.name}-instance"
  role = aws_iam_role.instance.name
}
