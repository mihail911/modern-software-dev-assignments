# Developer Automation Layer — Claude Code Guide

## Project Overview

This repo contains coursework that builds a unified **Developer Automation Layer**:

| Directory | What it is |
|-----------|-----------|
| `week3/` | Jikan (MyAnimeList) MCP Server (STDIO transport) |
| `week4/` | Claude Code agent — skills, subagents, hooks, MCP integration |

The two are integrated: the Week 3 MCP server powers the Week 4 agent's skills and subagents.

---

## Week 3 — Jikan (MyAnimeList) MCP Server

**Entry point**: `week3/server/main.js`  
**Requires**: nothing — Jikan is a public API, no token needed  
**Run**: `node server/main.js` (from `week3/`)

Tools exposed: `search_anime`, `get_anime`, `get_top_anime`, `search_manga`

---

## Week 4 — Claude Code Agent

**Agent guide**: `week4/CLAUDE.md`  
**Skills**: `.claude/skills/`  
**Subagents**: `.claude/agents/`  
**Hooks**: `.claude/settings.json`

---

## MCP Server Setup

Configure the Jikan MCP server in your Claude Code or Claude Desktop settings:

```json
{
  "mcpServers": {
    "jikan": {
      "command": "node",
      "args": ["server/main.js"],
      "cwd": "/absolute/path/to/week3"
    }
  }
}
```

Run the server manually: `cd week3 && npm install && node server/main.js`

---

## Safety Rules

- **Never** commit `.env` files or tokens
- **Never** hardcode secrets in any config file
