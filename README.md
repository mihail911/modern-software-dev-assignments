# Assignments for CS146S: The Modern Software Developer

This is the home of the assignments for [CS146S: The Modern Software Developer](https://themodernsoftware.dev), taught at Stanford University fall 2025.

## Repo Setup
These steps work with Python 3.12.

1. Install Anaconda
   - Download and install: [Anaconda Individual Edition](https://www.anaconda.com/download)
   - Open a new terminal so `conda` is on your `PATH`.

2. Create and activate a Conda environment (Python 3.12)
   ```bash
   conda create -n cs146s python=3.12 -y
   conda activate cs146s
   ```

3. Install Poetry
   ```bash
   curl -sSL https://install.python-poetry.org | python -
   ```

4. Install project dependencies with Poetry (inside the activated Conda env)
   From the repository root:
   ```bash
   poetry install --no-interaction
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
                      └── week3/server/main.py  (executes GitHub API calls)
```

### Key files

| Path | Purpose |
|------|---------|
| `.claude-plugin/plugin.json` | Plugin declaration and required user config (`GITHUB_TOKEN`, etc.) |
| `.claude-plugin/.mcp.json` | Wires Week 3 MCP server into Claude Code |
| `.claude/skills/sync-github/SKILL.md` | Defines `/sync-github` command logic |
| `.claude/settings.json` | Stop hook — auto-reviews task completion after every response |

See [CLAUDE.md](CLAUDE.md) for full details.