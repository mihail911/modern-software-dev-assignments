# Claude Code assignments

## Repo Setup

Requires **Node.js ≥ 18**.

1. Install dependencies for the Week 3 MCP server
   ```bash
   cd week3 && npm install
   ```

## Developer Automation Layer (Week 3 + Week 4)

Weeks 3 and 4 together build a unified **Developer Automation Layer** that integrates Claude Code with GitHub.

| Directory | Role |
|-----------|------|
| `week3/` | GitHub Issues MCP Server — the backend service that calls the GitHub API |
| `week4/` | Claude Code Plugin — the user-facing automation layer (skills, hooks, MCP wiring) |

### How they fit together

```
User: /sync-github owner/repo
    └── .claude/skills/sync-github/SKILL.md   (skill definition)
          └── calls MCP tools (list_issues, create_issue)
                └── .claude-plugin/.mcp.json   (MCP server connection config)
                      └── week3/server/main.js  (executes GitHub API calls)
```

### Key files

| Path | Purpose |
|------|---------|
| `.claude-plugin/plugin.json` | Plugin declaration and required user config (`GITHUB_TOKEN`, etc.) |
| `.claude-plugin/.mcp.json` | Wires Week 3 MCP server into Claude Code |
| `.claude/skills/sync-github/SKILL.md` | Defines `/sync-github` command logic |
| `.claude/settings.json` | Stop hook — auto-reviews task completion after every response |

See [CLAUDE.md](CLAUDE.md) for full details.


## Acknowledgements:
[themodernsoftware.dev](https://themodernsoftware.dev/)