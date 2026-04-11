---
name: sync-github
description: Sync incomplete action items from the week4 app to GitHub Issues using the GitHub MCP server. Triggered when user wants to push tasks to GitHub, sync action items, or create GitHub issues from the app.
context: fork
allowed-tools: Bash, Read
argument-hint: "[owner/repo]  e.g. my-org/week4-app"
---

# Sync Action Items → GitHub Issues

Sync all incomplete action items from the week4 Developer Command Center to GitHub Issues.
This skill calls the local app API and uses the GitHub MCP server to create issues.

## Steps

1. **Check the app is running**

```bash
curl -s http://localhost:8000/action-items | python3 -m json.tool 2>/dev/null || echo "APP_NOT_RUNNING"
```

If `APP_NOT_RUNNING`, tell the user to run `cd week4 && make run` first, then stop.

2. **Fetch incomplete action items**

```bash
curl -s http://localhost:8000/action-items
```

Filter to items where `completed == false`.

3. **Get target repo from arguments**

If `$ARGUMENTS` is provided (format: `owner/repo`), use it.
Otherwise ask the user: "Which GitHub repo should I sync to? (format: owner/repo)"

4. **Check existing issues to avoid duplicates**

Use the `list_issues` MCP tool:
- `owner`: parsed from repo argument
- `repo`: parsed from repo argument
- `state`: "open"

Collect existing issue titles (lowercase for comparison).

5. **Create issues for new action items**

For each incomplete action item not already in GitHub:
- Use `create_issue` MCP tool
- `title`: the action item description
- `body`: "Created automatically by the week4 Developer Command Center sync.\n\nAction item ID: {id}"
- `labels`: [] (no labels unless repo has them)

6. **Report results**

Print a summary table:
```
Created: N new issues
Skipped: M already existed
```
Include URLs of all newly created issues.

## Error Handling

- If MCP server is not available: tell the user to configure the GitHub MCP server in Claude Code settings (see CLAUDE.md)
- If `GITHUB_TOKEN` is missing: tell the user to set it in the MCP server env config
- If repo not found (404): tell the user to check the owner/repo spelling
