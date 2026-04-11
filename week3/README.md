# Week 3 — Jikan MCP Server

A Model Context Protocol (MCP) server that wraps the [Jikan REST API](https://jikan.moe/) (unofficial MyAnimeList API), exposing anime and manga data as MCP tools.

## Setup

Requires **Node.js ≥ 18**. No API key needed — Jikan is a public API.

```bash
npm install
node server/main.js
```

## Tools

| Tool | Description |
|------|-------------|
| `search_anime` | Search anime by title |
| `get_anime` | Get full details for an anime by ID |
| `get_top_anime` | Fetch top-ranked anime |
| `search_manga` | Search manga by title |

## Transport

STDIO — compatible with Claude Code and Claude Desktop MCP config.
