# Week 4 — Jikan Anime Agent

This week's deliverable is a Claude Code agent built on top of the Week 3 Jikan MCP server.

## What This Is

A Claude Code automation layer that lets users search, explore, and get recommendations for anime and manga — powered by the Jikan (MyAnimeList) MCP server from Week 3.

## Architecture

```
User request
  └── Skill (anime / recommend)          .claude/skills/
        └── Jikan MCP tools              week3/server/main.js
              search_anime · get_anime
              get_top_anime · search_manga
        └── Subagent (anime-researcher)  .claude/agents/
              └── Jikan MCP tools (same)

Hooks                                    .claude/settings.json
  └── Stop hook — completeness review
```

## Components

### Skills (`.claude/skills/`)

| Skill | Trigger | What it does |
|-------|---------|--------------|
| `anime` | `/anime <query>` | Search, browse, or get details for anime/manga |
| `recommend` | `/recommend <preference>` | Personalized recommendations via `anime-researcher` |

### Subagents (`.claude/agents/`)

| Agent | Purpose |
|-------|---------|
| `anime-researcher` | Deep-research agent — fetches full details for multiple titles and synthesizes a structured report |

### Hooks (`.claude/settings.json`)

| Hook | Event | Behaviour |
|------|-------|-----------|
| Stop review | `Stop` | Verifies all requested tasks were completed before Claude ends a response |

## MCP Server (Week 3)

The agent depends on the Jikan MCP server running from `week3/`.

**Start the server:**
```bash
cd week3 && npm install && node server/main.js
```

**Configure in Claude Code** (`~/.claude/mcp.json` or Claude Desktop config):
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

## Adding a New Skill

1. Create `.claude/skills/<name>/SKILL.md`
2. Add frontmatter: `name`, `description`, `context`, `allowed-tools`, `argument-hint`
3. Write step-by-step instructions in the body — reference Jikan MCP tool names
4. Test with `/<name>` in a Claude Code session

## Adding a New Subagent

1. Create `.claude/agents/<name>.md`
2. Add frontmatter: `name`, `description`, `tools` (list of allowed MCP tool names)
3. Write a focused system prompt — one clear responsibility per agent
4. Invoke from a skill using the agent's `name`

## Adding a New Hook

Edit `.claude/settings.json` under `hooks`. Available events:
- `PreToolUse` — before a tool runs
- `PostToolUse` — after a tool runs
- `Stop` — before Claude ends a response
