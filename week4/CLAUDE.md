# Week 4 — Claude Code Plugin

> Claude: this week's deliverable is a Claude Code plugin, not a backend app.

## What This Is

A reusable Claude Code plugin that automates developer workflows via skills, agents, hooks, and MCP integration. The plugin is installable by any developer who clones this repo.

## Plugin Structure

```
.claude-plugin/
├── plugin.json          # Plugin declaration (name, version, userConfig)
└── .mcp.json            # MCP server config (GitHub integration via Week 3)

.claude/
├── settings.json        # Hooks (Stop review)
└── skills/
    └── sync-github/
        └── SKILL.md     # Sync action items → GitHub Issues via MCP

week3/                   # MCP server that the plugin depends on
week4/
├── CLAUDE.md            # This file
├── docs/TASKS.md        # Plugin roadmap
└── writeup.md           # Assignment write-up
```

## Plugin Components

### Skills (`.claude/skills/`)
Reusable workflows invoked with `/skill-name` or natural language.

- **`sync-github`** — reads a data source and creates GitHub Issues via the Week 3 MCP server

### Hooks (`.claude/settings.json`)
Auto-triggered on Claude events — no user prompt needed.

- **Stop hook** — reviews session completeness before Claude finishes

### MCP Integration (`.claude-plugin/.mcp.json`)
Connects Claude Code to the Week 3 GitHub MCP server using `GITHUB_TOKEN`.

### Plugin Config (`.claude-plugin/plugin.json`)
Declares user-configurable values: `GITHUB_TOKEN`, `GITHUB_REPO`, `WEEK3_PATH`.

## Adding a New Skill

1. Create `.claude/skills/<name>/SKILL.md`
2. Add frontmatter: `name`, `description`, `allowed-tools`, `argument-hint`
3. Write step-by-step instructions in the body
4. Test by invoking `/<name>` in a Claude Code session

## Adding a New Agent

1. Create `.claude/agents/<name>.md`
2. Add frontmatter: `name`, `description`, `tools`, `model`
3. Write a focused system prompt — one responsibility per agent

## Adding a New Hook

Edit `.claude/settings.json` under `hooks`. Available events:
- `PreToolUse` — before a tool runs
- `PostToolUse` — after a tool runs
- `Stop` — before Claude ends a response
