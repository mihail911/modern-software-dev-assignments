# Claude Code assignments

## Repo Setup

Requires **Node.js ≥ 18**.

```bash
cd week3 && npm install
```

## Developer Automation Layer (Week 3 + Week 4)

Weeks 3 and 4 together build a unified **Developer Automation Layer** powered by the Jikan (MyAnimeList) API.

| Directory | Role |
|-----------|------|
| `week3/` | Jikan MCP Server — wraps the Jikan REST API, exposes 4 tools |
| `week4/` | Claude Code Agent — skills, subagents, and hooks built on top of the MCP server |

### How they fit together

```
User: /anime cowboy bebop
    └── .claude/skills/anime/SKILL.md     (skill definition)
          └── Jikan MCP tools             (search_anime, get_anime, …)
                └── week3/server/main.js  (Jikan REST API calls)

User: /recommend action completed
    └── .claude/skills/recommend/SKILL.md
          └── anime-researcher subagent   (.claude/agents/)
                └── Jikan MCP tools
```

### Key files

| Path | Purpose |
|------|---------|
| `week3/server/main.js` | Jikan MCP STDIO server |
| `.claude/skills/anime/SKILL.md` | `/anime` skill |
| `.claude/skills/recommend/SKILL.md` | `/recommend` skill |
| `.claude/agents/anime-researcher.md` | Deep-research subagent |
| `.claude/settings.json` | Stop hook |

See [CLAUDE.md](CLAUDE.md) for full details.

## Acknowledgements
[themodernsoftware.dev](https://themodernsoftware.dev/)
