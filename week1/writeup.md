# Week 1 Write-up

## Part I: Capture

**Setup** (enough for a reader to reproduce your capture):
```
claude --version:  2.1.119 (Claude Code)
mitmproxy version: 12.2.3 (Python 3.14.4, macOS-27.0-arm64)
proxy command:     mitmweb --listen-host 127.0.0.1 --listen-port 58888 \
                           --web-open-browser \
                           --mode reverse:https://api.anthropic.com \
                           -w session.flows
                   # run from ~/mitm-captures/ (outside any git repo)
settings file:     ~/Documents/GitHub/GAIA_TARA/.claude/settings.json
                   {
                     "env": {
                       "ANTHROPIC_BASE_URL": "http://127.0.0.1:58888",
                       "ENABLE_TOOL_SEARCH": "true"
                     },
                     "hooks": { ...pre-existing SessionStart hook preserved... }
                   }
```

**The session.** Add a shareable `/explorer/` database page (standalone, filterable table + charts backed by ~7.8k rows) to my GAIA_TARA repo — a static-site data-viz project already deploying to Netlify. Ran against `~/Documents/GitHub/GAIA_TARA`. Produced **37 `POST /v1/messages` requests** total; ~34 substantive turns after subtracting utility calls (`req_000/001` = handshake/`429`, `req_006` = session-title generation via a smaller-prompt call, `req_027` = compaction summary — see IV.e).

| Requirement | Evidence |
|---|---|
| Touched ≥ 2 files | 4 files mutated: `deliverable/src/explorer.html`,  `deliverable/src/doc.html` , `site/explorer/index.html` , `site/_probe.html`
| Failed at least once | `req_023.json messages[46]`: `is_error: true` from `mcp__claude-in-chrome__tabs_context_mcp` ("Multiple Chrome browsers are connected…"). Agent pivots at `req_024.json messages[47]` to headless Chrome via Bash. Full walk-through in Part IV.a. |
| Long enough to plan | Textual plan at `req_006.json messages[8].content[1]` (heading `## Plan: shareable database page at /explorer/`). Then 30+ turns of implementation, one hard compaction event at `req_027`. |
| Your own repo | `GAIA_TARA` (deployed at https://hyfincoredb.netlify.app/) — my own project |

**What you redacted** from the excerpts quoted below, and why:
> Nothing substantive. My user email (`asalecha@stanford.edu`) appears in the `<system-reminder>` quote in Part II.e — kept because it's my own institutional address on my own repo. My absolute home path (`/Users/aadeshsalecha/…`) is left visible for the same reason. GAIA_TARA is a private repo. No `x-api-key` / `authorization` headers appear anywhere in the writeup
## Part II: System Prompt Annotation


**a. Structure.**

- `system[0]` (70 chars): billing/telemetry header — `x-anthropic-billing-header: cc_version=2.1.283.2e7; cc_entrypoint=cli;`
- `system[1]` (57 chars): identity — `You are Claude Code, Anthropic's official CLI for Claude.`
- `system[2]` (10,600 chars): the main harness prompt annotated below.
- `messages[1]` with `role: "system"`: environment + deferred-tool list (see **d**).



Inside `system[2]` :
1. **Persona line** : "interactive agent that helps with software engineering tasks."
2. **Security posture (top-level `IMPORTANT`)** authorized-use bracket for security work, hard refusals for destructive/mass targeting. Placed first (so it overrides user requests below it)
3. **`# Harness`** — how the runtime works: markdown rendering, permission modes ("a denied call means the user declined it — adjust, don't retry verbatim"), hook semantics, `<pasted_content>` handling, parallel tool calls, `file:line` clickability.
4. **Code-style directive** — "Write code that reads like the surrounding code."
5. **Pronoun guidance** — default to they/them, never infer from a name.
6. **Irreversibility / faithful reporting** — confirm before hard-to-reverse actions, look before deleting, report outcomes without hedging.
7. **`# Session-specific guidance`** — `!`-prefix shell escape, `/skill` invocation rules.
8. **`# Memory`** — long spec for the file-based memory system (types, `MEMORY.md` index rules, verify-before-recommend).
9. **`# Environment`** — model IDs (Fable 5.1, Opus 5.5, Sonnet 5, Haiku 4.5), product surfaces, `/fast` note.
10. **`# Context management`** — auto-summarization on long convos, "when you have enough info, act."
11. **`EndConversation` note** — deferred tool with narrow trigger.
12. **`# Claude in Chrome browser automation`** — big optional capability block: how to batch-load deferred MCP tools, GIF recording, console debugging, alert avoidance, rabbit-hole guardrails, tab-context ritual.

**Why this order:** identity → non-negotiable safety → runtime mechanics → communication norms → memory → dynamic context handling → optional capabilities. Anything with refusal power (security, destructive-op gates) sits above capability grants (memory, browser), so a later capability can't be read as overriding an earlier prohibition.

**b. Tone and verbosity.** The controlling paragraphs are short but pointed:

```
Report outcomes faithfully: if tests fail, say so with the output; if a step was skipped, say
that; when something is done and verified, state it plainly without hedging.
```
```
When you have enough information to act, act. Do not re-derive facts already established in the
conversation, re-litigate a decision the user has already made, or narrate options you will not
pursue. If you are weighing a choice, give a recommendation, not an exhaustive survey.
```

> **Failure modes defended against:** (1) the "hedge and re-summarize" failure — models padding responses with restated context, option enumerations, and softening qualifiers, which burns tokens and hides whether the work actually landed. (2) the "false success" failure — declaring a task done without acknowledging skipped or failing steps.

**c. When not to act.** 

```
IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and
educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting,
supply chain compromise, or detection evasion for malicious purposes. Dual-use security tools
(C2 frameworks, credential testing, exploit development) require clear authorization context…
```
```
For actions that are hard to reverse or outward-facing, confirm first unless durably authorized
or explicitly told to proceed without asking; approval in one context doesn't extend to the next.
Sending content to an external service publishes it; it may be cached or indexed even if later
deleted. Before deleting or overwriting, look at the target.
```
```
Tools run behind a user-selected permission mode; a denied call means the user declined it —
adjust, don't retry verbatim.
```

> - **Security gate** (first quote):  carves out authorized security work rather than a blanket "no security tools." Buys usefulness in legitimate pentest/CTF contexts without opening a hole for mass-targeting requests.
> - **Destructive-op / scope gate** (second quote): Explicit "approval in one context doesn't extend to the next" prevents the model from treating a `git push` OK as an ambient `--force` license.
> - **Permission-mode gate** (third quote): defends against retry loops. Without it, a denied tool call gets the same tool re-fired with cosmetic changes.

**d. Environment context.** 
- Inside a `<system-reminder>` block in the user turn (`messages[0].content[0]`): user email, current branch, `git status` (dirty file list), and the last five commit messages.


- Inside a "system" -> "content", there was info about the OS version, primary working directory, scratchpad directory, 

**e. `<system-reminder>`.
```
<system-reminder>
As you answer the user's questions, you can use the following context:
# userEmail
The user's email address is asalecha@stanford.edu. Use it only to identify the user…
# gitStatus
Current branch: main
Status:
 M .claude/settings.json
 M .gitignore
 M deliverable/build.py
Recent commits:
099ae3e Add search_trends.csv results and ignore logs/…
…
IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to
this context unless it is highly relevant to your task.
</system-reminder>
```

```
<system-reminder>
Attribution for git commits and pull requests you create from here on (this replaces Claude
Code's own earlier attribution guidance, such as a previous copy of this reminder; the user's
own instructions about these lines, such as a CLAUDE.md or memory rule, take precedence over
this reminder, but do not add attribution lines this reminder leaves out):
- End git commit messages with:
Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
- End pull request descriptions with:
🤖 Generated with [Claude Code](https://claude.com/claude-code)
</system-reminder>
```

> **Where they appear:** only inside `user`-role message content, never in the top-level `system` field. They're `type: "text"` parts wrapped in the sentinel tag; the model is trained to treat them as system-controlled rather than user-authored probably.
>
> **Two distinct purposes evidenced:**
> 1. **Ephemeral context injection** — the first reminder is a *snapshot* of state (git status, recent commits, user email) that will be stale on the next turn. Wrapping it in `<system-reminder>` marks it as "background context, not user instructions" (per the harness's own memory guidance section) so the model doesn't try to act on it directly.
> 2. **Mutable policy override** — the second reminder literally says "this replaces Claude Code's own earlier attribution guidance." It's a versioned rule the harness can rewrite between turns.
>
> **Why mid-conversation rather than once up front:** the top-level `system` blocks are cache keys — anything that changes there invalidates the prompt cache for the whole conversation. Injecting via `<system-reminder>` in `messages` (a) lets the harness rev policies (attribution format) or refresh state (git status) *per turn* without cache invalidation, and (b) makes overrides *explicit* rather than requiring the model to detect contradictions with the static system prompt. The self-documenting "this replaces the previous copy of this reminder" phrasing tells us this for sure
>
## Part III: Tool Design Annotation

**Inventory.** Counts pulled from `req_002.json` (first substantive turn) and `req_022.json` (after mid-session change).

| Phase | Built-in (in `tools[]`) | MCP (in `tools[]`) | Deferred (in `role:"system"` msg) | **Total possible** |
|---|---|---|---|---|
| `req_002`–`req_021` | 16 | 0 | 59 (17 internal + 42 MCP-authenticate stubs) | **75** |
| `req_022`–`req_036` | 16 | **7** (all `mcp__claude-in-chrome__*`) | ~52 (7 chrome tools moved out of deferred into `tools[]`) | **~75** |

- **Built-in 16 (stable):** `Agent, Artifact, AskUserQuestion, Bash, DeferredToolPlaceholder, Edit, ListAgents, Read, ReportFindings, ScheduleWakeup, SendFeedback, ShareOnboardingGuide, Skill, ToolSearch, Workflow, Write`.
- **7 added at `req_022`:** `mcp__claude-in-chrome__` — `computer, javascript_tool, navigate, read_console_messages, resize_window, tabs_context_mcp, tabs_create_mcp`.
- **Deferred pool** (in the `role:"system"` message): 59 tool *names* — 17 internal (`Cron*`, `EnterPlanMode`, `WebFetch`, `WebSearch`, `Monitor`, `NotebookEdit`, `TaskStop`, etc.) and 42 `mcp__claude_ai_*__authenticate` / `__complete_authentication` stubs for third-party integrations I hadn't logged into (Slack, Figma, PubMed, bioRxiv, Airtable, etc.).

**Mid-session change.** At `req_022 messages[44]` has the `tools[]` array has grown by exactly the 7 `mcp__claude-in-chrome__*` names. [Speculative by me] Probably because it realized that it needed a browser to check the frontend changes, but switched to a headless browser later.


**Two tools.** `Bash` (the workhorse, 100+ calls) and `mcp__claude-in-chrome__tabs_context_mcp` (the MCP tool that fired *once* and returned an 800-char failure directive that is really the recovery instructions).

| | **Bash** | **mcp__claude-in-chrome__tabs_context_mcp** |
|---|---|---|
| **Required** | `command` | —  |
| **Optional** | `timeout`, `description`, `run_in_background`, `dangerouslyDisableSandbox` | `createIfEmpty` (bool) |
| **Not exposed** | `cwd`, stdin, env vars, interactive TTY (`-i` flags explicitly unsupported) | which browser to target, timeout, response filter, tab-group selection |
| **Description is defending against (quote → wrong behavior)** | *"Working directory persists between calls, but prefer absolute paths — `cd` in a compound command can trigger a permission prompt."* → models chaining `cd foo && …` and burning user consent prompts. *"Foreground `sleep` is blocked; use Monitor with an until-loop to wait on a condition."* → naive polling loops. *"Command output is displayed to you, not reliably to the user."* → the model treating Bash output as its way to *show* things instead of writing text. | *"CRITICAL: You must get the context at least once before using other browser automation tools so you know what tabs exist."* → the model calling `navigate`/`computer` blind and landing in someone else's tab. *"Each new conversation should create its own new tab (using tabs_create_mcp) rather than reusing existing tabs, unless the user explicitly asks."* → the model hijacking whatever the user was already reading. |
| **Deliberately does *not* do…** | No `cwd`, no stdin, no env override — implies the harness expects absolute paths and non-interactive invocations, so interactive tools are a category error, not a fallback. | Takes zero required arguments and doesn't ask *which* browser you mean. The whole failure recovery is stuffed into the *error string itself* (see IV.a below): an 800-char runbook telling you to call `AskUserQuestion` listing every browser. Implies the MCP designers preferred a fat error-payload over adding a `deviceId` parameter — probably because they wanted a general-purpose "unresolved ambient state" pattern reusable across MCP tools. |

**Why these two:** they sit at opposite ends of the tool-design spectrum. `Bash` is the general-purpose escape hatch and  `mcp__claude-in-chrome__tabs_context_mcp` is the opposite, very niche, does one specific thing, and also has a nice failure/recovery path.


## Part IV: Behavioral Analysis

**a. Error recovery**: `[OBSERVED]` · evidence: `req_023.json messages[45–46]` (failing tool_use + is_error tool_result), `req_024.json messages[47]` (recovery)

Failing call at `req_023 messages[45]`:
```json
{"name": "mcp__claude-in-chrome__tabs_context_mcp",
 "input": {"createIfEmpty": true}}
```
What the agent saw at `req_023 messages[46]` (`is_error: true`, verbatim, first ~400 chars):
```
Multiple Chrome browsers are connected to this account and none has been selected
for this session. Before any browser action, you MUST call the AskUserQuestion tool
with a question listing EVERY connected browser as a separate option (use the
display name as the label, and include the deviceId in parentheses), plus one
final option labeled exactly: "Open a confirmation screen in every connected Chrome
extension and let me select the right one there." Do not skip any connected browser
and do not pick one yourself…
```

> What it tried next, `req_024 messages[47]` (verbatim assistant `text` + first `tool_use`):
> ```
> Two Chrome browsers are connected, so rather than stop to ask which one, I'll
> test with headless Chrome from the shell.
> ```
> ```
> Bash: C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
>       "$C" --headless=new --disable-gpu --window-size=1400,1800
>            --screenshot=ex1.png --virtual-time-budget=4000
>            http://localhost:8765/explorer/ …
> ```
> **Turns to recover: 1.** the recovery tool call landed at `req_024`. 

**b. Planning**: `[OBSERVED]` · evidence: `req_006 messages[8].content[1]` (assistant `text`)

> Planning was **textual, reactive, and non-tool-based**. Across all 37 requests there is *zero* use of `TodoWrite`, `EnterPlanMode`, `ExitPlanMode`, `Workflow`, or `Agent` — all of which are in the inventory. Instead the agent produced a markdown plan directly as an assistant text block, headed `## Plan: shareable database page at /explorer/`, structured into Layout / Filter panel / Charts / Table / Default columns / Header hints. It was **elicited**: the user's initial prompt ends with *"Your plan should be brief and succinct not too long"* (`req_002 messages[0].content[2]`), and the plan appears at the first turn after the agent finished its read-only exploration (turns 3–5 were `cat build.py`, `head`, `wc`, `git diff`).

**c. Plans and task state**: `[OBSERVED]` · evidence: identical assistant text at `messages[8].content[1]` in every request `req_006` through `req_021` (16 turns)

> There is **no separate task-state channel**. The plan is a normal assistant `text` block persisted only because the whole message history persists; nothing echoes it as a `tool_result`, no `<system-reminder>` re-injects it, no summary appears in the harness prompt. Advancing happens implicitly — the agent reads its own earlier text and continues.

**d. Subagents**: `[INFERRED]` · evidence: zero `tool_use` entries with `name: "Agent"` in any of the 37 request files

> The `Agent` tool is declared in `tools[]` but never invoked. The probabl workload had a single linear thread (edit template → rebuild → screenshot → adjust) and no independent parallel workstreams, which is the usual delegation trigger.

**e. Context management**: `[OBSERVED]` · evidence: `len(messages)` and body size across all 37 requests

| Request | `len(messages)` | Body size (KB) | Notes |
|---|---|---|---|
| 002 | 2 | 143 | first substantive turn |
| 010 | 20 | ~250 | mid-plan implementation |
| 021 | 43 | ~830 | just before Chrome MCP loaded |
| 022 | 45 | ~880 | +7 chrome tools; user msg = `"Tool loaded."` |
| 023 | 46 | ~890 | failing chrome call |
| 026 | 53 | ~1000 | last turn before compaction |
| **027** | **1** | ~35 | **compaction event — single user msg** |
| 028 | 56 | ~1000 | resumed with fresh, cache-friendly history |
| 036 | 73 | 1,227 | final turn |

> Payloads grew **linearly through the session with one hard compaction event.** At `req_027` the message list drops from 53 messages to 1 — the sole content is a `<!-- Standalone database explorer → site/explorer/index.html … The data payload replaces the EXPLORER token below … -->` HTML comment. Then `req_028` resumes with 56 messages carrying the same task state forward. Reading this shape: the CLI ran a compaction (or `/compact`) between `req_026` and `req_028`, and `req_027`.  **Earlier tool_results are retained verbatim** through the session — the harness doesn't truncate individual results, it just periodically resets the whole tape.


## Part V: Reflection

**Two decisions you would copy**, and the problem each solves:
1. **`<system-reminder>`  in my normal conversations.** Now that I know this is what happens under the hood, what's stopping me from just typing it in my conversation turns to differenciate between env/config context I give it and the actual meat of the prompt. Because it's (probably) been trained to treat this as a system state object. 
2. **Errors that talk to the Agent.** When `tabs_context_mcp` failed at `req_023 msg[46]`, the `tool_result` was 800 chars of *what to do next*, not just "error." That lets tool authors ship rare/complex failure recovery without polluting the tool description that hits every request. The unhandled case gets handled *at the moment it happens*, which is exactly when the model is looking for guidance.

**One you would make differently** (engage with why it might be there):
> That same fat error payload told the model it "MUST call `AskUserQuestion`" — and my agent ignored it and pivoted to headless Chrome via Bash (see IV.a). The ceremonious "you must" framing invited a workaround. It's probably worded that way because the MCP designers were burned by models silently picking the wrong browser and disrupting a user's active tab. But the fix over-corrects!

**One thing the trace changed** about how you will steer a coding agent:
> Behavior is 90% harness, 10% prompt.  
  
My user prompt was a paragraph. If I want to change what the agent does on my repo, editing `.claude/settings.json` (env, hooks, permissions) and a project CLAUDE.md will pay off way more than tuning individual employee messages for mystartup. 


## Submission
1. `Command (⌘) + F` for `TODO`. No results means you're done.
  -- PS this isn't true because this line contains "TODO" hehe
2. Confirm no credentials or `x-api-key` headers made it into your quoted excerpts.
3. Push all changes to your remote repository and submit via Gradescope.
4. Don't forget to remove `ANTHROPIC_BASE_URL` from your repo's `.claude/settings.json`!
