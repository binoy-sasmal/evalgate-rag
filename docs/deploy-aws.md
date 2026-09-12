# Deploying evalgate-rag to AWS

Terraform for a single-instance deployment costing **~$13/month in eu-north-1**
— $0 on a legacy 12-month Free Tier account, or roughly 7 months of runway
against the $100 of credits on a current Free Tier plan. See [Cost](#cost) for
which applies to you. Everything lives in [`terraform/`](../terraform).

## What this builds, and what it deliberately does not

```
        your IP only (allowed_cidr)
                 │  :80
                 ▼
        ┌──────────────────────┐
        │  EC2 t3.micro        │   public subnet, Elastic IP
        │  ┌────────┐ ┌──────┐ │
        │  │  api   │ │  db  │ │   docker compose
        │  │ :8000  │ │ pg16 │ │   pgvector, named volume
        │  └────────┘ └──────┘ │
        └──────────────────────┘
                 │ egress via Internet Gateway (no NAT)
                 ▼
          Groq API · GHCR · SSM
```

**Not included, on purpose:**

| Omitted | Why |
|---|---|
| Application Load Balancer | ~$17/month, and free for only 12 months. One instance does not need one. |
| NAT Gateway | ~$33/month. A public subnet plus a closed security group gives the same egress for $0. |
| RDS | Free for 12 months, then ~$13/month. Postgres in a container on the same box is $0 forever. |
| Secrets Manager | $0.40/secret/month. SSM Parameter Store standard parameters are free. |
| TLS | Needs a domain name. The endpoint is IP-restricted instead — see [Hardening](#hardening). |
| Autoscaling, multi-AZ | One box, one AZ. This is a demo deployment, not an SLA. |

The tradeoff is honest: **this is a single instance with no redundancy.** If it
dies you lose the ingested database, and `terraform apply` rebuilds it in about
five minutes. Nothing here is stateful that isn't reproducible from the image.

## Prerequisites

1. **Terraform ≥ 1.6** and the **AWS CLI v2**, authenticated (`aws sts get-caller-identity`).
2. **A Groq API key** — <https://console.groq.com/keys>.
3. **Free Tier eligibility — check this first.** AWS changed the model in July
   2025: accounts created before then get the 12-month service allowances
   (750h EC2 t3.micro, 30GB EBS); newer accounts get signup credits on a
   6-month plan instead. Either works, but the bill differs. Confirm at
   **Billing and Cost Management → Free tier** in the console before applying.

## Deploy

### 1. Publish the image

`ci.yml` already pushes `ghcr.io/<you>/evalgate-rag:latest` on every push to
`main`. The instance pulls it with no registry credentials, so the package must
be **public** — a one-time setting at
`github.com/users/<you>/packages/container/evalgate-rag/settings`.

Push the deployment fixes to `main` first and let CI rebuild: the image needs
the baked-in embedding model and corpus, without which the container downloads
its model from HuggingFace on every start and cannot ingest at all.

To use a private registry or ECR instead, set `var.image` and add credentials
to `user_data.sh.tftpl`.

### 2. Store the API key

Out of band, so the secret never enters Terraform state:

```bash
aws ssm put-parameter \
  --name /evalgate-rag/llm-api-key \
  --type SecureString \
  --value "gsk_your_key_here" \
  --region eu-north-1
```

Terraform only grants the instance role permission to read this one parameter.

### 3. Configure and apply

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# set allowed_cidr to "$(curl -s ifconfig.me)/32" and your budget email
terraform init
terraform plan      # read this; confirm no NAT gateway, no load balancer, no RDS
terraform apply
```

First boot takes ~4 minutes: install Docker, pull the image, start Postgres,
ingest 125 documents into pgvector.

### 4. Verify

```bash
curl -s "$(terraform output -raw api_url)/health"    # {"status":"ok"}
curl -s "$(terraform output -raw api_url)/ready"     # {"status":"ready","chunks":452}
eval "$(terraform output -raw example_query)"
```

`/ready` reporting `chunks: 452` is the signal that ingest actually ran. A 503
with `store is empty` means it did not — the service would otherwise answer
every question with "I cannot answer this from the provided context", which
looks like a model failure rather than a missing deployment step.

If something is wrong, shell in without SSH:

```bash
eval "$(terraform output -raw shell)"        # SSM Session Manager
sudo tail -100 /var/log/evalgate-deploy.log
cd /opt/evalgate-rag && sudo docker compose logs --tail 50
```

## Updating

After CI publishes a new image:

```bash
eval "$(terraform output -raw shell)"
sudo evalgate-redeploy      # pull, restart, re-ingest, check readiness
```

Ingest is idempotent (`ON CONFLICT (doc_id, seq) DO UPDATE`), so this is safe to
re-run. Changing a Terraform variable that feeds `user_data` replaces the
instance outright — the box is cattle, and a rebuild re-ingests automatically.

## Tearing down

```bash
terraform destroy
```

Removes everything, the public IPv4 address included. **Do this whenever you
are not demoing.** Stopping the instance is not enough — EBS storage and the
public IPv4 address both keep billing while it sits stopped, which is the most
common way this stack quietly drains credits. Everything rebuilds with
`terraform apply` in about five minutes, corpus re-ingest included.

## Cost

**Check which Free Tier model your account is on first** (Billing → Free tier),
because they behave differently:

- **Legacy** (accounts before ~July 2025): 12 months of service allowances —
  750h/month EC2 t3.micro, 30GB EBS, 750h/month public IPv4. This stack bills
  **$0** for a year, then ~$13/month.
- **Current** (accounts after ~July 2025): ~$100 of signup credits on a 6-month
  plan, plus the "always free" services. There is no 750h EC2 allowance, so this
  stack draws roughly **$13/month against credits** — about 7 months of runway
  on $100, i.e. comfortably covered for the life of the plan.

Approximate monthly figures for eu-north-1, running continuously:

| Resource | Cost |
|---|---|
| EC2 t3.micro (730 h) | ~$7.90 |
| EBS gp3, 20GB | ~$1.60 |
| Public IPv4 address (730 h × $0.005) | ~$3.65 |
| VPC, IGW, security groups, IAM, SSM Parameter Store (standard) | $0 |
| Data transfer out (first 100GB/month) | $0 |
| **Total** | **~$13/month** |

**The public IPv4 charge is the one people miss.** Since February 2024 AWS bills
$0.005/hour for *every* public IPv4 address — Elastic or auto-assigned, instance
running or stopped. Releasing the EIP and relying on an auto-assigned address
saves nothing; only an IPv6-only deployment avoids it, which is not worth the
client-compatibility cost here.

A `$1` AWS Budget with both forecasted and actual alerts is created as a
tripwire, set deliberately low. On a credits plan this matters more than it
looks: the console shows credits draining rather than a bill arriving, so the
budget is what tells you the meter is running at all. Budgets notify, they do
not stop spend. If it fires unexpectedly, check for a second instance, an
orphaned EIP, or a leftover snapshot.

## Hardening

What you should change before this is anything more than a demo:

- **`allowed_cidr` is the only access control.** There is no authentication on
  `/query`. Every call spends Groq quota against a 1K-request/day free tier
  shared with the eval gate, so opening this to `0.0.0.0/0` lets a stranger
  take your CI gate down. If you need public access, put an API key check in
  front of `/query` first.
- **No TLS.** Traffic is plain HTTP. Add Caddy as a third container with a real
  domain for automatic Let's Encrypt certificates — budget ~50MB of RAM, which
  is significant on a 1GB box.
- **No backups.** The pgvector volume is not snapshotted. This is deliberate:
  the corpus ships in the image and ingest rebuilds the database from scratch,
  so the data is reproducible rather than precious.
- **Single AZ, single instance.** Any instance replacement is downtime.

## Memory

1GB of RAM is the binding constraint, and it is why several choices look
unusual:

- `user_data` provisions **2GB of swap**. Without it, ingest — which holds the
  ONNX session (~350MB) and embeds 450+ chunks while Postgres is also running —
  gets something OOM-killed, usually Postgres, which presents as database
  corruption rather than as a sizing problem.
- Postgres runs with `shared_buffers=64MB` and `max_connections=20` instead of
  the pg16 defaults. The corpus is ~450 rows; the defaults buy nothing and cost
  RAM the API needs.
- The image sets `OMP_NUM_THREADS=1` so ONNX does not spin a thread per core
  and contend with Postgres for burst credits.

If you move to a larger instance, all three can be relaxed.
