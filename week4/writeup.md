# Week 4 Write-up

## SUBMISSION DETAILS

Name: **Tingwei** \
SUNet ID: **TODO** \
Citations: claude.nagdy.me/learn/, anthropic.com/engineering/claude-code-best-practices, docs.anthropic.com/en/docs/claude-code/sub-agents

This assignment took me about **8** hours to do.


## YOUR RESPONSES

---

### Automation #1: CLAUDE.md Guidance Files

**a. Design inspiration**

> From the Claude Code best-practices guide: *"CLAUDE.md is automatically read when starting a conversation, letting you provide repo-specific instructions that influence Claude's behavior."* I wanted a single source of truth that tells Claude how the plugin is structured and how it should behave — without having to explain the layout in every conversation. Two files were created: one at the repo root covering both weeks, and one at `week4/CLAUDE.md` with plugin-specific detail.

**b. Design of the automation**

> - **Goal**: Give Claude persistent context so it doesn't need to be re-briefed every session.
> - **Inputs**: None — Claude reads the file automatically on startup.
> - **Outputs**: Claude immediately knows the plugin structure, what components exist, how to add new skills/agents/hooks, and what the safety constraints are.
> - **Content**:
>   - Plugin directory layout with component-level annotations
>   - How to add a new skill, agent, or hook
>   - Safety guardrails (never commit tokens, never skip hooks)
>   - Available automations and their trigger phrases

**c. How to run it**

> No explicit run step needed. Claude Code reads `CLAUDE.md` automatically when opening a session in this directory. To verify it's loaded, start a session and ask: *"How do I add a new skill to this plugin?"* — Claude should answer correctly without any additional context.
>
> To update behavior, edit the relevant `.md` file. Changes apply on the next session start.
>
> **Rollback**: Remove or rename `CLAUDE.md`. No side effects.

**d. Before vs. after**

> **Before**: Every new conversation required explaining the plugin structure, what components exist, and how to extend it. This consumed the first few hundred tokens of every session.
>
> **After**: Claude arrives pre-briefed. The first useful action can be "add a skill that does X" — no setup needed.

**e. How the automation enhanced the developer workflow**

> With CLAUDE.md in place, Claude correctly understood that:
> - New skills go in `.claude/skills/<name>/SKILL.md` with specific frontmatter
> - Agents go in `.claude/agents/<name>.md`
> - Hooks are configured in `.claude/settings.json`, not in skill files
>
> This prevented confusion between skill invocation and hook triggers — two concepts that look similar but work differently. The CLAUDE.md file acts as the single source of truth for the entire plugin architecture.

---

### Automation #2: `sync-github` Skill + MCP Integration

**a. Design inspiration**

> From claude.nagdy.me/learn/skills: *"Skills are reusable capabilities that Claude discovers and uses automatically based on context... supporting progressive loading, dynamic shell context injection, subagent isolation, and invocation control."*
>
> The `sync-github` skill was designed to be the integration point between the Week 3 MCP server and Week 4's automation layer — turning a manual multi-step GitHub workflow into a single command.

**b. Design of the automation**

> **`sync-github`** (`.claude/skills/sync-github/SKILL.md`)
> - **Goal**: Push a list of items to GitHub Issues via the Week 3 MCP server — no browser, no copy-paste.
> - **Inputs**: Optional `owner/repo` argument. Falls back to asking the user.
> - **Steps**: Fetch source data → call `list_issues` to check duplicates → call `create_issue` for each new item → report results.
> - **Isolation**: `context: fork` — runs in a separate subagent so it doesn't pollute main context.
> - **Trigger**: Automatically detected when user says "sync to GitHub", or explicitly via `/sync-github`.
>
> **MCP wiring** (`.claude-plugin/.mcp.json`)
> - Points Claude Code at the Week 3 STDIO server
> - Uses `${WEEK3_PATH}` and `${GITHUB_TOKEN}` from `plugin.json` userConfig — no hardcoded paths or secrets

**c. How to run it**

> ```
> /sync-github my-org/my-repo
> ```
> or just say: *"Sync to my GitHub repo"*
>
> **Prerequisites**: GitHub MCP server must be configured with a valid `GITHUB_TOKEN`. The `.claude-plugin/.mcp.json` handles wiring automatically when the plugin is installed.
>
> **Rollback**: GitHub issues can be closed manually. The skill skips items that already have an issue with the same title, so re-running is safe (idempotent).

**d. Before vs. after**

> **Before**:
> 1. Manually copy each item description
> 2. Go to GitHub, click "New issue" for each one
> 3. Paste title, write a body, submit
> 4. Repeat N times — with no deduplication
>
> **After**: One command syncs everything, checks for duplicates, and reports results with issue URLs.

**e. How the automation enhanced the developer workflow**

> `sync-github` connected the Claude Code plugin layer directly to GitHub Issues, making it possible to manage project tracking without leaving the editor. The MCP integration means Claude can call `create_issue` and `list_issues` as native tools — the same way it calls file editing tools. This turns GitHub from an external website into a callable service within a Claude Code session.

---

### Automation #3: Plugin Packaging + Stop Hook

**a. Design inspiration**

> From the Claude Code plugin spec: plugins allow bundling skills, agents, hooks, and MCP config into a single installable unit with user-configurable variables. The Stop hook concept came from observing that Claude often finishes a response before fully verifying its own work — a lightweight review prompt at session end catches gaps before the user has to point them out.

**b. Design of each automation**

> **Plugin packaging** (`.claude-plugin/plugin.json`)
> - **Goal**: Make the entire automation suite installable by any developer, with configurable secrets.
> - **userConfig**: Three variables — `GITHUB_TOKEN` (sensitive), `GITHUB_REPO`, `WEEK3_PATH` — so the plugin works in any environment without editing source files.
> - **Effect**: Another developer can install this plugin, fill in their token and repo, and immediately have `sync-github` available with their own credentials.
>
> **Stop hook** (`.claude/settings.json`)
> - **Goal**: Before Claude ends any response, review whether the original request was fully addressed.
> - **Type**: `prompt` — runs a review prompt through Claude itself, not a shell command.
> - **Behaviour**: Returns non-zero exit code if work is incomplete, triggering Claude to continue. Returns zero if complete or if waiting for user input (to avoid false positives on legitimate conversation pauses).

**c. How to run it**

> **Plugin**: Install by adding the plugin to Claude Code settings. Fill in `GITHUB_TOKEN`, `GITHUB_REPO`, and `WEEK3_PATH` when prompted.
>
> **Stop hook**: Runs automatically at the end of every Claude response. No user action required.
>
> **Rollback**: Remove the `Stop` entry from `settings.json` to disable the hook. Remove `plugin.json` to unpackage.

**d. Before vs. after**

> **Before (plugin)**:
> Every developer who wanted these automations had to manually copy skill files, configure hooks, and set up the MCP server path by hand — error-prone and undocumented.
>
> **After**: One plugin install with three config values covers everything.
>
> **Before (Stop hook)**:
> Claude would sometimes finish a response having addressed only part of a multi-part question, leaving the user to notice and re-ask.
>
> **After**: The hook catches incomplete responses automatically. If Claude missed a step, it continues rather than stopping.

**e. How the automation enhanced the developer workflow**

> The plugin packaging transforms a collection of ad-hoc config files into a distributable developer tool. A new team member can install the plugin, provide their GitHub token, and immediately have the full workflow available — no README-following required.
>
> The Stop hook changed the quality of responses in this session itself: it caught several cases where the conversation was mid-task and prompted continuation rather than an early stop. The hook's "waiting for user input is not a gap" clause prevents it from firing on legitimate conversation pauses.
