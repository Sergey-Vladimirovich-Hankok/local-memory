# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |

## Reporting a vulnerability

**Do not open a public issue.** Use the private vulnerability reporting
mechanism of the hosting platform:

1. Go to the repository's **Security** tab on GitHub.
2. Click **Report a vulnerability**.
3. Describe the issue, its impact, and (if possible) a minimal reproduction.

We aim to acknowledge reports within 7 days and to ship a fix as soon as
practically possible.

## Scope notes

- local-memory is a local-first tool: the database, indexes, and embedding
  caches live on your machine. The project does not collect telemetry and does
  not call any service unless you explicitly configure `EMBED_API_URL`.
- Anything you ingest into the database is stored as-is, in plain SQLite. If
  your conversations are sensitive, protect the `MEMORY_DB_PATH` file with
  filesystem permissions and backups accordingly.

## Out of scope

- Vulnerabilities in third-party dependencies (report them upstream).
- Misuse of the tool to store data you are not allowed to store.
