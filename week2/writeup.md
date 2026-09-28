# Week 2 Write-up

## Part I: The Server

**API chosen**, and why:
> TODO

**How to run it** (one command):
```
TODO
```

| Tool | What it does | Read/Write | Composes with |
|---|---|---|---|
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |


## Part II: Agent Ergonomics

For each, point at the code (`file:line`) and say what it buys.

| Decision | Where | Why |
|---|---|---|
| Schema-level constraint | TODO | TODO |
| Output shaping (fields kept vs. dropped) | TODO | TODO |
| Structured errors (retry vs. don't-retry) | TODO | TODO |
| Docstring that chains tools together | TODO | TODO |
| Brake on the write tool | TODO | TODO |

**One thing you changed after watching the agent misuse a tool:**
> TODO


## Part III: OAuth

**Flow**: how a token is obtained, cached, and refreshed:
> TODO

**Scopes requested**, and why each is necessary:
> TODO

**Secrets**: what's in env, what's gitignored:
> TODO

**Token dies mid-session**: what the agent sees:
> TODO


## Part IV: Integration

**Registration config** (`.mcp.json.example`) and the client you used:
> TODO

**End-to-end transcript**: the prompt, the tools that fired with their arguments, the result:
```
TODO
```

**A failure, handled**: what you provoked, what the agent saw, what it did next:
```
TODO
```

**Protocol-level test**: what it covers and how to run it:
> TODO


## Submission
1. `Command (⌘) + F` for `TODO`. No results means you're done.
2. Confirm no tokens, client secrets, cached token file, or real `.mcp.json` are committed.
3. Push all changes to your remote repository and submit via Gradescope.
4. Clean up (optional): remove the server from your agent config, delete your cached token, and revoke the OAuth app's access.
