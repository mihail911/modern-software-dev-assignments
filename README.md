# Claude Code Assignments

## Overview

Weeks 3 and 4 together build a unified **Developer Automation Layer** powered by the [Jikan REST API](https://jikan.moe/) (unofficial MyAnimeList API).

| Directory | Role |
|-----------|------|
| [week3/](week3/) | Jikan MCP Server — wraps the Jikan REST API, exposes 4 tools via MCP |
| [week4/](week4/) | Claude Code Agent — skills and subagents built on top of the MCP server |

## Prerequisites

- **Node.js ≥ 18**
- No external API key required — Jikan is a free public API

## Project Structure

```
.
├── week3/                   # MCP server
│   ├── server/
│   │   └── main.js          # MCP server entry point (STDIO transport)
│   ├── __tests__/
│   │   └── week3.test.js    # Integration tests (live Jikan API)
│   ├── package.json
│   └── README.md
├── week4/                   # Claude Code agent layer
│   ├── CLAUDE.md            # Architecture documentation
│   ├── __tests__/
│   │   └── week4.test.js    # Structure / frontmatter tests
│   ├── package.json
│   └── README.md
└── .claude/
    ├── skills/              # User-invocable skills (anime, recommend)
    ├── agents/              # Subagents (anime-researcher)
    └── settings.json        # Hooks configuration
```

## Week 3 — Jikan MCP Server

An MCP server exposing anime and manga data over STDIO, compatible with Claude Code and Claude Desktop.

**Exposed tools:**

| Tool | Description |
|------|-------------|
| `search_anime` | Search anime by title |
| `get_anime` | Get full details for an anime by MAL ID |
| `get_top_anime` | Fetch the current top-ranked anime list |
| `search_manga` | Search manga by title |

**Run the server:**

```bash
cd week3
npm install
npm start          # node server/main.js
```

## Week 4 — Claude Code Agent

A Claude Code automation layer on top of the Week 3 MCP server.

**Components:**

| Type | Name | Description |
|------|------|-------------|
| Skill | `anime` | Search, browse, or get details for anime/manga |
| Skill | `recommend` | Personalized recommendations via the `anime-researcher` subagent |
| Subagent | `anime-researcher` | Deep-research agent for multi-title synthesis |

Skills live in `.claude/skills/`, agents in `.claude/agents/`. See [week4/CLAUDE.md](week4/CLAUDE.md) for architecture details.

## Running Tests

**Week 3** — integration tests, makes live Jikan API calls:

```bash
cd week3 && npm install && npm test
```

**Week 4** — structural tests, checks file layout and frontmatter:

```bash
cd week4 && npm install && npm test
```

## Autograding

Grades are calculated automatically when you open a Pull Request to `master`:

1. Push your work to a branch and open a PR targeting `master`
2. GitHub Actions runs the test suite for each week
3. A comment is posted on the PR with the breakdown and total score (out of 90)

| Week | Max Score | What is tested |
|------|-----------|----------------|
| Week 3 | 100 pts | Functionality (50), reliability (30), code quality (20) |
| Week 4 | 100 pts | Functionality (75), code quality (25) |

Each week is graded independently — week3 and week4 each produce a separate score comment.

## Acknowledgements
[themodernsoftware.dev](https://themodernsoftware.dev/)
