# Week 1 Write-up

## Part I: Capture

**Setup** (enough for a reader to reproduce your capture):
```
claude --version:  TODO
mitmproxy version: TODO
proxy command:     TODO
settings file:     TODO (path + env block)
```

**The session.** What task, against what repo, and how many `POST /v1/messages` requests did it produce?
> TODO

| Requirement | Evidence |
|---|---|
| Touched ≥ 2 files | TODO |
| Failed at least once | TODO |
| Long enough to plan | TODO |
| Your own repo | TODO |

**What you redacted** from the excerpts quoted below, and why:
> TODO


## Part II: System Prompt Annotation

The top-level `system` field is a **list of 3 text blocks**, followed by a `role: "system"` message *inside* `messages[1]`. All annotations below cite `req_002.json` (first substantive turn; identical structure repeats in `req_003.json`).

- `system[0]` (70 chars): billing/telemetry header — `x-anthropic-billing-header: cc_version=2.1.283.2e7; cc_entrypoint=cli;`
- `system[1]` (57 chars): identity — `You are Claude Code, Anthropic's official CLI for Claude.`
- `system[2]` (10,600 chars): the main harness prompt annotated below.
- `messages[1]` with `role: "system"`: environment + deferred-tool list (see **d**).

**a. Structure.** Major sections of `system[2]` in order.

1. **Persona line** — one sentence: "interactive agent that helps with software engineering tasks." Anchors the role before any rules load.
2. **Security posture (top-level `IMPORTANT`)** — authorized-use bracket for security work, hard refusals for destructive/mass targeting. Placed first so any downstream instruction is read through this filter.
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

> **Failure modes defended against:** (1) the "hedge and re-summarize" failure — models padding responses with restated context, option enumerations, and softening qualifiers, which burns tokens and hides whether the work actually landed. (2) the "false success" failure — declaring a task done without acknowledging skipped or failing steps. Together they push toward terse, load-bearing prose: an assertion of outcome, not a narration of process.

**c. When not to act.** Three distinct gates, each buying a different guarantee:

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

> - **Refusal gate** (first quote): domain-scoped refusal — carves out authorized security work rather than a blanket "no security tools." Buys usefulness in legitimate pentest/CTF contexts without opening a hole for mass-targeting requests.
> - **Destructive-op / scope gate** (second quote): buys resistance to the "one blanket approval → many silent destructive follow-ups" failure. Explicit "approval in one context doesn't extend to the next" prevents the model from treating a `git push` OK as an ambient `--force` license. The publish-is-forever line addresses paste-to-pastebin-style leaks.
> - **Permission-mode gate** (third quote): defends against retry loops. Without it, a denied tool call gets the same tool re-fired with cosmetic changes; the instruction reframes denial as *user signal* rather than *transient error*.

**d. Environment context.** Machine/repo/session info is split across **two locations, both outside the top-level `system` list**:

- Inside a `role: "system"` **message** at `messages[1]` (observed in `req_003.json`): CWD (`/Users/aadeshsalecha/Documents/GitHub/GAIA_TARA`), platform/OS/shell, session-specific scratchpad path (`/private/tmp/claude-501/…`), exact model ID (`claude-opus-5-5[1m]`), knowledge cutoff, and the full list of deferred-tool names available via `ToolSearch`.
- Inside a `<system-reminder>` block in the user turn (`messages[0].content[0]`): user email, current branch, `git status` (dirty file list), and the last five commit messages.

> The split is deliberate. The static `system[*]` blocks are **cache-friendly** — identical across every request in the conversation so Anthropic's prompt cache can hit. Anything session-specific (CWD, model routing, tool availability) or *turn-specific* (git status) lives in `messages`, where it's cheap to vary without invalidating the cached system prefix. That's also why `role: "system"` shows up as a message here at all: it's a system-authored instruction that's allowed to change, unlike the immutable `system` prefix.

**e. `<system-reminder>`.** Both examples observed in `req_003.json` `messages[0].content` (the first user turn, wrapping the actual user message):

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

> **Where they appear:** only inside `user`-role message content, never in the top-level `system` field. They're `type: "text"` parts wrapped in the sentinel tag; the model is trained to treat them as system-controlled rather than user-authored.
>
> **Two distinct purposes evidenced:**
> 1. **Ephemeral context injection** — the first reminder is a *snapshot* of state (git status, recent commits, user email) that will be stale on the next turn. Wrapping it in `<system-reminder>` marks it as "background context, not user instructions" (per the harness's own memory guidance section) so the model doesn't try to act on it directly.
> 2. **Mutable policy override** — the second reminder literally says "this replaces Claude Code's own earlier attribution guidance." It's a versioned rule the harness can rewrite between turns.
>
> **Why mid-conversation rather than once up front:** the top-level `system` blocks are cache keys — anything that changes there invalidates the prompt cache for the whole conversation. Injecting via `<system-reminder>` in `messages` (a) lets the harness rev policies (attribution format) or refresh state (git status) *per turn* without cache invalidation, and (b) makes overrides *explicit* rather than requiring the model to detect contradictions with the static system prompt. The self-documenting "this replaces the previous copy of this reminder" phrasing is the giveaway — the mechanism is designed for mutation.


## Part III: Tool Design Annotation

**Inventory.** Did the set change across requests? If so, what triggered it?

| Built-in | MCP | Deferred | **Total** | Changed mid-session? |
|---|---|---|---|---|
| TODO | TODO | TODO | **TODO** | TODO |

**Two tools.** Pick tools that differ from each other.

| | Tool 1 | Tool 2 |
|---|---|---|
| Name | TODO | TODO |
| Key schema fields | TODO | TODO |
| Required vs. optional vs. not exposed, and why | TODO | TODO |
| Description is defending against… (quote + the wrong behavior) | TODO | TODO |
| Deliberately does *not* do… and what that implies | TODO | TODO |

Why these two?
> TODO


## Part IV: Behavioral Analysis

**Every answer must be labeled `[OBSERVED]` or `[INFERRED]` and cite its evidence. Unlabeled answers earn no credit.**

**a. Error recovery**: `TODO: label` · evidence: `TODO`

What the agent saw, verbatim:
```
TODO
```
What it tried next, and turns to recover:
> TODO

**b. Planning**: `TODO: label` · evidence: `TODO`
> TODO

**c. Plans and task state**: `TODO: label` · evidence: `TODO` \
How does one get created and advanced? What does the model see about task state each turn, and where does it live in the request:
> TODO

**d. Subagents**: `TODO: label` · evidence: `TODO` \
When the agent delegates, what the subagent is told, and what comes back:
> TODO

**e. Context management**: `TODO: label` · evidence: `TODO` \
What changed in the payloads as the session grew:
> TODO


## Part V: Reflection

**Two decisions you would copy**, and the problem each solves:
1. TODO
2. TODO

**One you would make differently** (engage with why it might be there):
> TODO

**One thing the trace changed** about how you will steer a coding agent:
> TODO


## Submission
1. `Command (⌘) + F` for `TODO`. No results means you're done.
2. Confirm no credentials or `x-api-key` headers made it into your quoted excerpts.
3. Push all changes to your remote repository and submit via Gradescope.
4. Don't forget to remove `ANTHROPIC_BASE_URL` from your repo's `.claude/settings.json`!
