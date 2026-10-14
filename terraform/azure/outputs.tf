locals {
  api_url = "https://${azurerm_container_app.api.ingress[0].fqdn}"
  rg      = azurerm_resource_group.this.name
}

output "api_url" {
  description = "Base URL of the API. Reachable only from var.allowed_cidr."
  value       = local.api_url
}

output "run_ingest" {
  description = "Step 1 after the first apply: load the corpus into PostgreSQL."
  value       = "az containerapp job start -g ${local.rg} -n ${azurerm_container_app_job.ingest.name}"
}

output "restart_api" {
  description = "Step 2: restart so the API rebuilds its in-process BM25 index from the ingested corpus."
  value       = "az containerapp revision restart -g ${local.rg} -n ${azurerm_container_app.api.name} --revision ${azurerm_container_app.api.latest_revision_name}"
}

output "readiness_check" {
  description = "Readiness: database reachable and corpus ingested. Returns 503 with a reason if not."
  value       = "curl -s ${local.api_url}/ready"
}

output "example_query" {
  description = "End-to-end smoke test."
  value       = <<-EOT
    curl -s ${local.api_url}/query -X POST -H 'content-type: application/json' \
      -d '{"question": "What are the maximum fines for prohibited AI practices?"}'
  EOT
}

output "logs" {
  description = "Follow the API container's logs."
  value       = "az containerapp logs show -g ${local.rg} -n ${azurerm_container_app.api.name} --container api --follow"
}

output "application_insights" {
  description = "Where the traces and metrics land (Azure portal > Application Insights)."
  value       = azurerm_application_insights.this.name
}
