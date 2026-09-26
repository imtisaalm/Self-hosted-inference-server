# Security

- Tool names and schemas are configured by the application, not by model output.
- Sensitive agent tools require explicit approval.
- SQLite files and trace logs are plaintext; protect them like application data.
- The streaming proxy binds to loopback and requires a bearer token.
- Python execution uses Docker with network, filesystem, process, CPU, memory, and output limits.
- Docker is a shared-kernel boundary. Use a VM-based sandbox for hostile multi-tenant code.
- Guardrail patterns are a first pass, not an authorization layer.
