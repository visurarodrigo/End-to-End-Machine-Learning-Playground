# Utils - Shared Utilities & Helpers

General-purpose helper functions used across the application. Utilities are small, focused, and independent of specific features.

## Design Philosophy

Utilities should be:
- **Generic:** Work across multiple contexts
- **Focused:** Do one thing well
- **Stateless:** Pure functions with no side effects
- **Lightweight:** Minimal dependencies
- **Reusable:** No endpoint-specific logic

## Common Utility Patterns

- Data validation (CSV format, file types, numeric columns)
- Data transformation (standardization, normalization, scaling)
- Formatting utilities (round metrics, format outputs)
- Parsing helpers (extract metadata, convert types)

Add utilities here as needed across the app to avoid code duplication.
