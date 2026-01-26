---
name: review
description: Review code changes for quality issues
context: fork
agent: Explore
---

# Code Review

Review code changes or specified files for quality issues.

## Usage

- `/review` - Review uncommitted changes
- `/review src/file.py` - Review specific file
- `/review --staged` - Review staged changes only

## Review Checklist

Analyze code for:

### Type Safety
- [ ] Type hints on all function signatures
- [ ] Proper Optional handling (no bare None returns without annotation)
- [ ] Correct use of Union types

### Error Handling
- [ ] Exceptions caught at appropriate boundaries
- [ ] User-facing errors have clear messages
- [ ] External service failures handled gracefully

### Security
- [ ] No path traversal vulnerabilities (user input in file paths)
- [ ] Input validation on API endpoints
- [ ] No hardcoded credentials or secrets

### Code Style
- [ ] Functions under 50 lines
- [ ] Using pathlib instead of os.path
- [ ] Dataclasses/Pydantic for structured data
- [ ] No unnecessary complexity
- [ ] Using MetadataManager for run metadata (not raw JSON)
- [ ] Using PipelineConfig for complex parameter passing

### Test Coverage
- [ ] New functions have corresponding tests
- [ ] Edge cases covered
- [ ] External services mocked

## Output Format

Provide specific file:line references for each issue found:

```
src/example.py:42 - Missing type hint for return value
src/example.py:78 - Catching bare Exception, should be more specific
tests/test_example.py - Missing test for error case
```

$ARGUMENTS