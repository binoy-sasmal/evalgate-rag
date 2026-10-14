# One user-assigned identity is the only credential the workload has. It is
# granted exactly two things, each next to the resource it applies to: calling
# the model (openai.tf) and reading one secret (database.tf). User-assigned
# rather than system-assigned so the role assignments can exist before the
# container app does, instead of racing its first start.
resource "azurerm_user_assigned_identity" "app" {
  name                = "${var.name}-identity"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  tags                = local.tags
}
