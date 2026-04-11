# Developer Automation Layer — Claude Code Guide

## Project Overview

This repo contains coursework that builds a unified **Developer Automation Layer**:

| Directory | What it is |
|-----------|-----------|
| `week3/` | GitHub Issues MCP Server (STDIO transport) |
| `week4/` | Claude Code plugin — skills, hooks, MCP integration |

The two are integrated: the Week 3 MCP server powers the `sync-github` skill in Week 4.

---

## Week 3 — GitHub MCP Server

**Entry point**: `week3/server/main.py`  
**Requires**: `GITHUB_TOKEN` environment variable  
**Run**: `GITHUB_TOKEN=<token> python -m server.main` (from `week3/`)

Tools exposed: `get_repo_info`, `list_issues`, `create_issue`, `close_issue`

---

## Week 4 — Claude Code Plugin

**Plugin declaration**: `.claude-plugin/plugin.json`  
**MCP config**: `.claude-plugin/.mcp.json`  
**Skills**: `.claude/skills/`  
**Hooks**: `.claude/settings.json`

### Plugin layout

```
.claude-plugin/
├── plugin.json          # Plugin name, version, userConfig (GITHUB_TOKEN etc.)
└── .mcp.json            # MCP server wiring for Week 3

.claude/
├── settings.json        # Hooks (Stop review)
└── skills/
    └── sync-github/
        └── SKILL.md     # Sync items → GitHub Issues

week4/
├── CLAUDE.md            # Week 4 plugin guide
├── docs/TASKS.md        # Plugin roadmap
└── writeup.md           # Assignment write-up
```

### `.claude-plugin/` — Plugin Declaration & MCP Wiring

| File | Purpose |
|------|---------|
| `plugin.json` | Declares plugin name, version, and required user config (`GITHUB_TOKEN`, `GITHUB_REPO`, `WEEK3_PATH`) |
| `.mcp.json` | Tells Claude Code how to launch the Week 3 MCP server (command, cwd, env vars) |

When the plugin is installed, Claude Code automatically mounts the MCP server, making `list_issues` and `create_issue` tools available to skills.

### `.claude/skills/` — User-Invocable Skills

Skills are custom commands defined in `.claude/skills/<name>/SKILL.md`. Currently there is one:

**`sync-github`** — syncs incomplete action items from the week4 app to GitHub Issues.

Key properties:
- `context: fork` — runs in an isolated sub-agent, keeping the main conversation clean
- `allowed-tools: Bash, Read` — restricted to only these two tools

Execution flow:
1. Check the week4 app is running (`localhost:8000`)
2. Fetch incomplete action items from the API
3. Call `list_issues` via MCP to check for existing issues (deduplication)
4. Call `create_issue` via MCP for each new item
5. Report a summary table with URLs of newly created issues

### `.claude/settings.json` — Hooks

Contains a **Stop hook** that fires automatically at the end of every Claude response. It runs a prompt to verify all requested tasks were completed — if any gaps are found, it returns a non-zero exit code so Claude continues working.


---

## MCP Server Setup

The `.claude-plugin/.mcp.json` configures the connection automatically when the plugin is installed. Set these values in your plugin config:

- `GITHUB_TOKEN` — GitHub Personal Access Token with `repo` scope
- `WEEK3_PATH` — absolute path to the `week3/` directory
- `GITHUB_REPO` — target repo in `owner/repo` format

---

## Safety Rules

- **Never** commit `.env` files or tokens
- **Never** hardcode `GITHUB_TOKEN` in any config file — use the plugin userConfig
