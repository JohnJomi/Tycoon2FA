"""Ingest package: Gmail retrieval, RFC-822 parsing, and normalization.

See ARCHITECTURE.md section 3.

- `gmail_client` - transport and OAuth. Gmail API -> base64url raw bytes.
- `parser`       - MIME, headers, bodies, URLs, attachments -> ParsedEmail.
- `pipeline`     - the boundary the rest of the system consumes: paginates a
                   Gmail query, dedupes, and yields IngestedMessage.
"""
