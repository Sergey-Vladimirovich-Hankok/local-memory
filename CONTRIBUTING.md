# Contributing

Thanks for considering a contribution to local-memory.

## Ground rules

- Local-first is the product promise: no cloud services, no mandatory network
  access, no telemetry. Do not add dependencies that break this.
- Python 3.10+ standard library + numpy + the official `mcp` SDK. Keep the
  core light; put optional extras (scikit-learn) behind `[semantic]`.
- Result row format is a public contract:
  `{session_id, position, content, score, context}`. Changing it is a breaking
  change and needs a discussion in an issue first.

## Setup

```bash
git clone <repo-url> local-memory
cd local-memory
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[semantic]"
```

## Workflow

1. Create a branch from `main`.
2. Make your change; keep functions small and error handling at the edges
   (tools and CLI never crash the transport or the shell).
3. Add or update tests in `tests/`. Fixtures must stay synthetic — real
   conversation data must never enter the repository.
4. Run the checks:

   ```bash
   python3 -m pytest tests/ -q
   python3 -m py_compile src/local_memory/*.py tests/*.py
   ```

5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open a pull request against `main`.

## Code style

- Follow PEP 8; 4-space indent; type hints on public functions.
- Docstrings on public functions: one-line summary + Args/Returns where useful.
- No secrets, no absolute home paths, no personal identifiers in code, docs,
  or tests.

## Security

See [SECURITY.md](SECURITY.md). If you find a vulnerability, do not open a
public issue — use the private advisory process.
