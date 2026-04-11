# Developer Automation Layer — Claude Code Guide

## Project Overview

This repo contains coursework that builds a unified **Developer Automation Layer**:

| Directory | What it is |
|-----------|-----------|
| `week3/` | GitHub Issues MCP Server (STDIO transport) |
| `week4/` | Claude Code plugin — skills, hooks, MCP integration |

The two are integrated: the Week 3 MCP server powers the `sync-github` skill in Week 4.

---

## Week 3 — Jikan (MyAnimeList) MCP Server

**Entry point**: `week3/server/main.js`  
**Requires**: nothing — Jikan is a public API, no token needed  
**Run**: `node server/main.js` (from `week3/`)

Tools exposed: `search_anime`, `get_anime`, `get_top_anime`, `search_manga`

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

**`anime`** — search and explore anime/manga via the Jikan MCP server.

Key properties:
- `context: fork` — runs in an isolated sub-agent, keeping the main conversation clean
- `allowed-tools: Bash` — restricted to shell calls only

Execution flow:
1. Parse `$ARGUMENTS` to determine intent (search, get by ID, top list, or manga)
2. Call the appropriate Jikan MCP tool (`search_anime`, `get_anime`, `get_top_anime`, or `search_manga`)
3. Format and display results as a table (lists) or detail card (single item)

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
