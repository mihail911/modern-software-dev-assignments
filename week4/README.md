# Week 4 — The Autonomous Coding Agent IRL

A Claude Code automation layer built on top of the Week 3 Jikan MCP server, enabling anime/manga search and personalized recommendations.

## Components

| Type | Name | Description |
|------|------|-------------|
| Skill | `anime` | Search, browse, or get details for anime/manga |
| Skill | `recommend` | Personalized recommendations via `anime-researcher` subagent |
| Subagent | `anime-researcher` | Deep-research agent for multi-title synthesis |

See [CLAUDE.md](CLAUDE.md) for architecture details.
