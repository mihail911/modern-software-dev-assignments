# Week 4 — The Autonomous Coding Agent IRL

Build at least **2 automations** using any combination of the following Claude Code features:

- Skills (custom slash commands in `.claude/skills/`)
- `CLAUDE.md` files for repository guidance
- Subagents (role-specialized agents in `.claude/agents/`)
- MCP servers integrated into Claude Code

## Tasks

1. Build **2 or more automations** that meaningfully improve a workflow.
2. Use the Week 3 MCP server as the data source for at least one automation.
3. Include at least one **skill** and one **subagent**.
4. Configure at least one **hook** in `.claude/settings.json`.
5. Document the agent architecture in `week4/CLAUDE.md`.

## Automation Types

### A) Skills (`.claude/skills/<name>/SKILL.md`)
Reusable workflows invoked by name. Use `$ARGUMENTS` for inputs, declare `allowed-tools` in frontmatter, and keep steps focused and idempotent.

### B) `CLAUDE.md` guidance files
Automatically read at session start. Use for architecture context, run commands, safety guardrails, and workflow conventions.

### C) Subagents (`.claude/agents/<name>.md`)
Specialized agents with a focused system prompt, specific tools, and a single responsibility. Skills can delegate to subagents for complex multi-step work.

### D) Hooks (`.claude/settings.json`)
Auto-triggered on Claude events (`PreToolUse`, `PostToolUse`, `Stop`) — no user prompt needed.

## Evaluation Rubric (90 pts)

| Category | Points | Criteria |
|----------|--------|----------|
| Functionality | 35 | 2+ automations working end-to-end with the MCP server |
| Design | 20 | Clear separation of concerns between skills, subagents, and hooks |
| Documentation | 20 | `CLAUDE.md` explains architecture; skills have clear instructions |
| Code Quality | 15 | Focused prompts, correct frontmatter, idempotent steps |
| Extra Credit | +10 | +5 parallel subagents · +5 additional hook type |
