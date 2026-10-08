# Project Rules — Ask the EU AI Act

These rules apply to all work in this repository.

## Tech stack
- Python 3.12, FastAPI, pytest for the backend.
- Next.js + TypeScript for the UI.

## Configuration & secrets
- Never hardcode secrets. Read all config from environment variables.
- Keep `.env` gitignored; commit a `.env.example` documenting every variable.

## Code quality
- Keep the code small and readable. No unused abstractions or features.
- Every module gets tests.
- Never claim metrics you haven't actually run.
- Explain each non-obvious design choice in a short code comment or in `design.md`.

## Workflow
- Work one task at a time and stop for review after each.
