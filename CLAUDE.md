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

### Layout

```
.claude/
├── settings.json        # Hooks (Stop review)
├── skills/
│   ├── anime/SKILL.md   # Search and browse anime/manga
│   └── recommend/SKILL.md  # Personalized recommendations
└── agents/
    └── anime-researcher.md  # Deep-research subagent

week4/
├── CLAUDE.md            # Agent architecture guide
└── assignment.md        # Assignment tasks and rubric
```

### Skills (`.claude/skills/`)

| Skill | Trigger | Description |
|-------|---------|-------------|
| `anime` | `/anime <query>` | Search, browse, or look up anime/manga by title or ID |
| `recommend` | `/recommend <preference>` | Personalized recommendations using the `anime-researcher` subagent |

### Subagents (`.claude/agents/`)

**`anime-researcher`** — deep-research agent that makes multiple Jikan MCP calls and synthesizes a structured report. Invoked by the `recommend` skill for multi-title research.

### Hooks (`.claude/settings.json`)

Contains a **Stop hook** that fires automatically at the end of every Claude response. It runs a prompt to verify all requested tasks were completed — if any gaps are found, it returns a non-zero exit code so Claude continues working.

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
