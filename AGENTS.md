# Repository AGENTS.md

This repository prefers simple, pipeline-oriented Python code.

## Style

- Keep script `main()` functions readable and focused on the overall flow.
- Extract a function only when logic is repeated or already forms one clear work unit.
- Prefer small functions and data flow over adding classes or heavy abstraction.
- Use config-driven behavior for experiment settings instead of hard-coded branches when practical.
- Keep naming consistent for model variants, dataset fields, and output artifacts.
- Comments should be in English and only added when they help explain non-obvious logic.

## Error Handling

- Fail fast on invalid config, unknown model names, missing files, or unexpected data shape.
- Raise clear errors with enough context to debug the issue.
- Do not hide errors with broad fallback behavior unless the user explicitly asks for it.

## Editing

- Match the existing structure and patterns in nearby files before introducing a new style.
- Prefer minimal changes that solve the current task without broad refactors.
