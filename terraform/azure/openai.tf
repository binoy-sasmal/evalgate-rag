resource "azurerm_cognitive_account" "openai" {
  name                = "${local.unique}-openai"
  resource_group_name = azurerm_resource_group.this.name
  location            = coalesce(var.openai_location, var.location)
  kind                = "OpenAI"
  sku_name            = "S0"
  tags                = local.tags

  # Token authentication needs a resource-specific hostname; the shared
  # regional endpoint only accepts keys.
  custom_subdomain_name = "${local.unique}-openai"

  # No API keys at all. This is what makes "never put a secret in Terraform
  # state" hold for the model: a cognitive account's keys are attributes of the
  # resource, so with key access enabled they would sit in state whether or not
  # anything here read them. The API authenticates as its managed identity.
  local_auth_enabled = false
}

resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.openai_model.name
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = var.openai_model.name
    version = var.openai_model.version
  }

  sku {
    name     = var.openai_model.sku
    capacity = var.openai_capacity
  }
}

# Inference only: this role can call deployments but cannot create, change or
# delete them.
resource "azurerm_role_assignment" "openai_user" {
  scope                = azurerm_cognitive_account.openai.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}
