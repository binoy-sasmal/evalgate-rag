# Tripwire, not a cost control: an Azure budget notifies, it does not stop
# spend. Unlike the AWS stack this one is not expected to be free -- the
# PostgreSQL server bills by the hour -- so the threshold is the expected
# monthly cost plus headroom, and crossing it means an assumption broke.
resource "azurerm_consumption_budget_resource_group" "guard" {
  name              = "${var.name}-guard"
  resource_group_id = azurerm_resource_group.this.id
  amount            = var.budget_amount
  time_grain        = "Monthly"

  time_period {
    # Must be the first of a month. Computed at creation, then left alone.
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", timestamp())
  }

  # Forecasted: warns while there is still time to terraform destroy.
  notification {
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Forecasted"
    contact_emails = [var.budget_notification_email]
  }

  # Actual: catches what the forecast misses early in a month.
  notification {
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Actual"
    contact_emails = [var.budget_notification_email]
  }

  lifecycle {
    ignore_changes = [time_period]
  }
}
