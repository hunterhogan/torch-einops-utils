# Development goals

1. Code changes should not require lucidrains, the developer, to change their coding style.
   1. They don't use type annotations very often, so the code for typing should be unobtrusive.
2. Confirm that changes do not break packages that import these symbols.
3. Use formatters and linters to make new code match the style of the existing codebase.
