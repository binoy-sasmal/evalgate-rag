# The database password never enters Terraform state.
#
# An ephemeral resource exists only for the duration of a run, and the two
# places the value goes -- the server's admin password and a Key Vault secret
# -- take it through write-only arguments, which Terraform sends to the API
# and then forgets. State records only db_password_version. A plain
# random_password resource, or a `sensitive = true` variable, would both be
# plaintext in terraform.tfstate.
#
# The ephemeral value is regenerated on every run but only *sent* when the
# version changes, so both destinations always receive the same password in
# the same apply. Rotation is: bump db_password_version, apply.
ephemeral "random_password" "db" {
  length = 32
  # The API builds a postgresql:// URL from this, and URL-reserved characters
  # in a password would need escaping it does not do.
  special = false
}

resource "azurerm_postgresql_flexible_server" "this" {
  name                = "${local.unique}-pg"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  version             = "16"
  sku_name            = var.postgres_sku
  storage_mb          = 32768
  tags                = local.tags

  administrator_login               = "rag"
  administrator_password_wo         = ephemeral.random_password.db.result
  administrator_password_wo_version = var.db_password_version

  backup_retention_days         = 7
  public_network_access_enabled = true

  lifecycle {
    # Azure picks the availability zone when none is given, and reports it
    # back; without this every later plan proposes to move the server.
    ignore_changes = [zone]
  }
}

resource "azurerm_postgresql_flexible_server_database" "rag" {
  name      = "rag"
  server_id = azurerm_postgresql_flexible_server.this.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# pgvector ships with Flexible Server but is not creatable until allow-listed;
# without this the API's `CREATE EXTENSION IF NOT EXISTS vector` is refused.
resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.this.id
  value     = "VECTOR"
}

# KNOWN WEAKNESS. The 0.0.0.0 rule is Azure's "allow access from Azure
# services": it admits connections from any Azure-hosted address, in any
# tenant, not just this container app. They still face TLS and the password,
# but the port is reachable. It is here because a Consumption-plan Container
# Apps environment has no stable outbound address to allow-list. The real fix
# is a VNet-integrated environment with the server on private access, listed
# under Hardening in docs/deploy-azure.md.
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.this.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_key_vault" "this" {
  # Vault names are global and capped at 24 characters.
  name                = "${trimsuffix(substr(var.name, 0, 14), "-")}-${random_string.suffix.result}-kv"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  tags                = local.tags

  rbac_authorization_enabled = true
  soft_delete_retention_days = 7
  # Off so `terraform destroy` followed by a fresh apply works. A vault that
  # held anything irreplaceable would turn this on.
  purge_protection_enabled = false
}

# Whoever runs Terraform may write secrets; the workload may only read them.
resource "azurerm_role_assignment" "kv_deployer" {
  scope                = azurerm_key_vault.this.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_role_assignment" "kv_app" {
  scope                = azurerm_key_vault.this.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_key_vault_secret" "db_password" {
  name             = "db-password"
  key_vault_id     = azurerm_key_vault.this.id
  value_wo         = ephemeral.random_password.db.result
  value_wo_version = var.db_password_version

  depends_on = [azurerm_role_assignment.kv_deployer]
}
