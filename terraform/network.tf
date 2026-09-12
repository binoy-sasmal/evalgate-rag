# A minimal public-subnet VPC.
#
# There is deliberately no private subnet and no NAT Gateway. The instance
# needs outbound internet (GHCR pull, Groq API calls) and a NAT Gateway is
# ~$33/month plus data processing -- on its own, several times the cost of
# everything else here combined, and the single largest reason "small" AWS
# deployments produce surprising bills. A public subnet with an Internet
# Gateway gives the same egress for $0; ingress is closed by the security
# group rather than by network topology.

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "this" {
  cidr_block           = "10.20.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = var.name }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = var.name }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.20.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = { Name = "${var.name}-public" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = { Name = "${var.name}-public" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "instance" {
  name        = "${var.name}-instance"
  description = "evalgate-rag API ingress"
  vpc_id      = aws_vpc.this.id

  tags = { Name = var.name }
}

# No SSH rule. Shell access is via SSM Session Manager (see iam.tf), which
# needs no inbound port, no key pair, and no bastion -- and leaves an audit
# trail. Port 22 open to the internet on a box holding an API key is not worth
# the convenience.
resource "aws_vpc_security_group_ingress_rule" "api" {
  security_group_id = aws_security_group.instance.id
  description       = "API, restricted to var.allowed_cidr"
  cidr_ipv4         = var.allowed_cidr
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.instance.id
  description       = "Outbound: GHCR image pull, Groq API, SSM endpoints, package updates"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
