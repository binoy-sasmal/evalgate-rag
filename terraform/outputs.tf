output "api_url" {
  description = "Base URL of the API. Reachable only from var.allowed_cidr."
  value       = "http://${aws_eip.this.public_ip}"
}

output "health_check" {
  description = "Liveness: the process is up."
  value       = "curl -s http://${aws_eip.this.public_ip}/health"
}

output "readiness_check" {
  description = "Readiness: database reachable and corpus ingested. Returns 503 with a reason if not."
  value       = "curl -s http://${aws_eip.this.public_ip}/ready"
}

output "example_query" {
  description = "End-to-end smoke test."
  value       = <<-EOT
    curl -s http://${aws_eip.this.public_ip}/query -X POST -H 'content-type: application/json' \
      -d '{"question": "What are the maximum fines for prohibited AI practices?"}'
  EOT
}

output "shell" {
  description = "Shell in over SSM Session Manager (no SSH key, no open port 22)."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.this.id}"
}

output "deploy_log" {
  description = "First-boot provisioning log, once you are on the box."
  value       = "sudo tail -f /var/log/evalgate-deploy.log"
}

output "instance_id" {
  value = aws_instance.this.id
}
