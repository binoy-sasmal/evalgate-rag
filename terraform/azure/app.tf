locals {
  db_env = {
    DB__HOST            = azurerm_postgresql_flexible_server.this.fqdn
    DB__PORT            = "5432"
    DB__USER            = azurerm_postgresql_flexible_server.this.administrator_login
    DB__NAME            = azurerm_postgresql_flexible_server_database.rag.name
    EMBEDDING__PROVIDER = "fastembed"
  }

  api_env = merge(local.db_env, {
    # The v1 endpoint is OpenAI-compatible, so the existing client works
    # unchanged; `model` is the deployment name.
    LLM__BASE_URL = "${azurerm_cognitive_account.openai.endpoint}openai/v1"
    LLM__MODEL    = azurerm_cognitive_deployment.chat.name
    LLM__AUTH     = "azure_managed_identity"
    # Tells the identity endpoint which identity to mint a token for.
    AZURE_CLIENT_ID = azurerm_user_assigned_identity.app.client_id

    TELEMETRY__ENABLED       = "true"
    TELEMETRY__OTLP_ENDPOINT = "http://localhost:4318" # the sidecar below
  })
}

resource "azurerm_container_app" "api" {
  name                         = "${var.name}-api"
  resource_group_name          = azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.this.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  # Resolved from Key Vault by the platform, using the identity. Terraform
  # passes a reference, never the value.
  secret {
    name                = "db-password"
    key_vault_secret_id = azurerm_key_vault_secret.db_password.versionless_id
    identity            = azurerm_user_assigned_identity.app.id
  }

  # An ingestion address rather than a credential in the usual sense: it
  # permits sending telemetry to this one resource and nothing else. It is an
  # attribute of the Application Insights resource, so it is in state.
  secret {
    name  = "appinsights-connection-string"
    value = azurerm_application_insights.this.connection_string
  }

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }

    # /query has no authentication; this is the only thing in front of it.
    ip_security_restriction {
      name             = "allowed-cidr"
      action           = "Allow"
      ip_address_range = var.allowed_cidr
    }
  }

  template {
    min_replicas = var.min_replicas
    # One replica: the BM25 index is built in process at startup, and this is
    # a demo deployment, not an SLA.
    max_replicas = 1

    container {
      name   = "api"
      image  = var.image
      cpu    = 0.5
      memory = "1Gi"

      dynamic "env" {
        for_each = local.api_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name        = "DB__PASSWORD"
        secret_name = "db-password"
      }

      # Startup allows 150 s for the embedding model to load and the BM25
      # index to be read from Postgres before liveness takes over.
      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 5
        failure_count_threshold = 30
      }

      # /health, not /ready, on purpose. /ready fails until the corpus has
      # been ingested, which is an operator step after the first apply; wired
      # to the platform it would mark a correctly running revision unhealthy
      # instead of returning its own 503 that says what to do.
      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
    }

    container {
      name   = "otel-collector"
      image  = "otel/opentelemetry-collector-contrib:0.115.1"
      cpu    = 0.25
      memory = "0.5Gi"
      args   = ["--config=env:OTEL_COLLECTOR_CONFIG"]

      env {
        name  = "OTEL_COLLECTOR_CONFIG"
        value = file("${path.module}/otel-collector.yaml")
      }

      env {
        name        = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        secret_name = "appinsights-connection-string"
      }
    }
  }

  depends_on = [
    # The identity must be able to read the secret and call the model before
    # the first replica starts, and the API creates its schema at boot.
    azurerm_role_assignment.kv_app,
    azurerm_role_assignment.openai_user,
    azurerm_postgresql_flexible_server_configuration.extensions,
    azurerm_postgresql_flexible_server_firewall_rule.azure_services,
  ]
}

# Ingest is a job, not part of the API's startup: with scale-to-zero an init
# container would re-embed the whole corpus on every cold start. Run it once
# after the first apply and again after a corpus or chunking change.
resource "azurerm_container_app_job" "ingest" {
  name                         = "${var.name}-ingest"
  resource_group_name          = azurerm_resource_group.this.name
  location                     = azurerm_resource_group.this.location
  container_app_environment_id = azurerm_container_app_environment.this.id
  tags                         = local.tags

  replica_timeout_in_seconds = 1800
  replica_retry_limit        = 1

  manual_trigger_config {
    parallelism              = 1
    replica_completion_count = 1
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  secret {
    name                = "db-password"
    key_vault_secret_id = azurerm_key_vault_secret.db_password.versionless_id
    identity            = azurerm_user_assigned_identity.app.id
  }

  template {
    container {
      name    = "ingest"
      image   = var.image
      cpu     = 0.5
      memory  = "1Gi"
      command = ["python", "scripts/ingest.py"]

      dynamic "env" {
        for_each = local.db_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name        = "DB__PASSWORD"
        secret_name = "db-password"
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.kv_app,
    azurerm_postgresql_flexible_server_configuration.extensions,
    azurerm_postgresql_flexible_server_firewall_rule.azure_services,
  ]
}
