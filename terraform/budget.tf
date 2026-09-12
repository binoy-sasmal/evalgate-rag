# Tripwire, not a cost control: AWS Budgets notifies, it does not stop spend.
# This stack should bill $0 while the account is Free Tier eligible, so a $1
# actual spend means an assumption broke -- and you want that as an email in a
# day, not as a surprise at the end of the month.
resource "aws_budgets_budget" "guard" {
  name         = "${var.name}-guard"
  budget_type  = "COST"
  limit_amount = "1.0"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Track cost BEFORE credits are applied.
  #
  # This is what makes the alarm mean anything on a credits-based Free Tier
  # plan. By default Budgets nets promotional credits off the cost, so an
  # account running $13/month entirely on signup credits reports $0 spend --
  # and a $1 budget would never fire while the credit balance silently drained
  # to zero, at which point real charges start with no warning at all.
  # Excluding credits makes this alarm report the meter rather than the
  # invoice, which is the thing worth knowing early.
  cost_types {
    include_credit = false
  }

  # Forecasted: warns before the money is spent, while there is still time to
  # terraform destroy.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_notification_email]
  }

  # Actual: forecasting is unreliable in the first days of a month and on brand
  # new accounts, so this one catches what the forecast misses.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_notification_email]
  }
}
