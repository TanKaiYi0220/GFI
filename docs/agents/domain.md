# Domain Docs

How engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root.
- `docs/adr/` for architectural decisions that touch the area being changed.

If any of these files do not exist, proceed silently. The producer skill creates them lazily when terms or decisions need to be recorded.

## File structure

This is a single-context repo:

```text
/
+-- CONTEXT.md
+-- docs/adr/
+-- src/
```

## Use the glossary's vocabulary

When output names a domain concept, use the term as defined in `CONTEXT.md`. Do not drift to synonyms the glossary explicitly avoids.

If the concept is not in the glossary yet, either reconsider the term or note the gap for a docs-backed design conversation.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly instead of silently overriding it.
