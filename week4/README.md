# Week 4 — The Autonomous Coding Agent IRL

A Claude Code automation layer built on top of the Week 3 Jikan MCP server, enabling anime/manga search and personalized recommendations.

## Automations

| Type | Name | Trigger | Description |
|------|------|---------|-------------|
| Skill | `anime` | `/anime <query>` | Search, browse, or get details for anime/manga |
| Skill | `recommend` | `/recommend <preference>` | Personalized recommendations via `anime-researcher` subagent |
| Subagent | `anime-researcher` | invoked by `recommend` | Deep-research agent — fetches full details for multiple titles and synthesizes a structured report |
| Hook | Stop review | `Stop` event | Verifies all requested tasks were completed before Claude ends a response |

## Architecture

See [CLAUDE.md](CLAUDE.md) for full architecture details and component descriptions.
