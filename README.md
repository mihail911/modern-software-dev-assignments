# Claude Code assignments

## Repo Setup

Requires **Node.js ≥ 18**.

1. Install dependencies for the Week 3 MCP server
   ```bash
   cd week3 && npm install
   ```

## Developer Automation Layer (Week 3 + Week 4)

Weeks 3 and 4 together build a unified **Developer Automation Layer**.

| Directory | Role |
|-----------|------|
| `week3/` | Jikan (MyAnimeList) MCP Server — wraps the Jikan REST API, no auth required |
| `week4/` | Claude Code Plugin — the user-facing automation layer (skills, hooks, MCP wiring) |

### How they fit together

```
User: invokes an MCP-aware client (Claude Desktop, Cursor, etc.)
    └── .claude-plugin/.mcp.json   (MCP server connection config)
          └── week3/server/main.js  (Jikan API calls)
```

### Key files

| Path | Purpose |
|------|---------|
| `.claude-plugin/plugin.json` | Plugin declaration and required user config |
| `.claude-plugin/.mcp.json` | Wires Week 3 MCP server into Claude Code |
| `.claude/skills/sync-github/SKILL.md` | Defines `/sync-github` command logic |
| `.claude/settings.json` | Stop hook — auto-reviews task completion after every response |

See [CLAUDE.md](CLAUDE.md) for full details.


## Acknowledgements:
[themodernsoftware.dev](https://themodernsoftware.dev/)