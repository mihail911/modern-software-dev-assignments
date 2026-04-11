# Week 4 — Jikan Anime Agent

A Claude Code automation layer built on top of the Week 3 Jikan MCP server, enabling anime/manga search and personalized recommendations.

## Architecture

```
User request
  └── Skill (anime / recommend)          .claude/skills/
        └── Jikan MCP tools              week3/server/main.js
        └── Subagent (anime-researcher)  .claude/agents/

```

## MCP Server (Week 3)

The agent depends on the Jikan MCP server running from `week3/`. See the root `CLAUDE.md` for setup instructions.
