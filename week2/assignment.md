# Week 2: Build an MCP Server for a Third-Party API

## Assignment Overview

Build an MCP server that exposes a third-party API to your coding agent. The API work is the easy part. What is graded is **agent ergonomics**: whether a model can pick the right tool, call it correctly, and recover when it doesn't. Plus a working **OAuth** flow, because most APIs worth wrapping don't take a static key.

### Learning Goals

- **Design** tool interfaces for a consumer that only ever sees your schemas and descriptions.
- **Implement** OAuth token acquisition, caching, and refresh inside a long-running server.
- **Observe** how your design choices change what the agent actually does.

## Materials

- **[FastMCP](https://github.com/jlowin/fastmcp)**: what you'll build with.
- **[MCP specification](https://modelcontextprotocol.io/)**: tool annotations, transports, error semantics.
- **[Reference MCP servers](https://github.com/modelcontextprotocol/servers)**: read one before writing yours.

## Part I: Build the Server (25 pts)

Pick an API that uses **OAuth 2.0**: GitHub, Google (Gmail/Calendar/Drive), Notion, Linear, Slack, Spotify, and Strava all qualify. Pick something you actually use; you will be a better judge of what the tools should do.

Requirements:

- **At least 3 tools**, and they must **compose**: the output of one feeds another (e.g. `search_x` returns IDs that `get_x(id)` accepts).
- **At least one write tool** that mutates real state.
- Code lives in `week2/`, runs over stdio, and starts from a single documented command.

## Part II: Agent Ergonomics (30 pts)

This is the assignment. Every choice below is about the fact that the agent sees your schemas and docstrings and *nothing else*.

- **Constraints in the schema, not the prose.** `sort: Literal["stars", "forks"]` beats `sort: str` plus a docstring sentence. If the schema and the docstring ever disagree, the agent gets validation failures it cannot explain.
- **Shape your output.** Do not pass raw API JSON through. Pick the fields an agent needs. Raw payloads are token-heavy and couple your contract to someone else's API.
- **Errors are data, not tracebacks.** Return something structured and actionable, and include enough detail for the agent to distinguish *don't retry* (bad ID) from *retry* (rate limit, network blip). A traceback ends the agent's turn; `{"error": "rate limited, retry after 42s"}` doesn't.
- **Docstrings are the contract.** Say where an ID comes from ("the numeric ID from `list_templates`") so tools chain. Use `FastMCP(instructions=...)` for workflow guidance that spans tools.
- **Write tools need a brake.** Use tool annotations (`readOnlyHint` on readers), and give the irreversible tool either a `dry_run` parameter or a preview-then-commit split.

## Part III: OAuth (25 pts)

A real OAuth flow, not a pasted bearer token.

- Full authorization-code exchange, with **cached** tokens and **silent refresh** on expiry.
- **Minimal scopes.** Ask for read-only if you only read, and justify each scope in the writeup.
- **No secrets in the repo.** Read from env from the first commit. Commit `.mcp.json.example`; gitignore the real one.
- **Handle the token dying mid-session.** A long-running server must not try to open a browser during a tool call. Return an actionable error instead.

## Part IV: Integrate and Demonstrate (20 pts)

- Register the server with Claude Code or Cursor and commit the config (`.mcp.json.example`).
- **Show one end-to-end transcript** where the agent chains at least two of your tools to answer a real prompt: the prompt, which tools fired with what arguments, and the result.
- **Show one failure** and how your error design handled it. Provoke it if you have to (bad ID, revoked token).
- At least one test that exercises the server **through the MCP protocol**, not just by calling your Python functions.

## Deliverables

In `week2/`: your server code, `.mcp.json.example`, tests, and a completed `writeup.md`.

## Evaluation Rubric (100 pts total)

| Part | Points | What earns full credit |
|---|---|---|
| I. Server | 25 | 3+ composing tools, one write tool, runs from a documented command |
| II. Agent ergonomics | 30 | Schema-level constraints, shaped output, structured errors, chaining docstrings, a brake on the write tool |
| III. OAuth | 25 | Working code exchange with caching and refresh; minimal justified scopes; no secrets committed; sane mid-session expiry path |
| IV. Integration | 20 | Committed config, a real chained transcript, a demonstrated failure path, one protocol-level test |

Deductions for committed secrets, and for tools whose docstrings contradict their schemas.

## SUBMISSION INSTRUCTIONS

1. Make sure you have all changes pushed to your remote repository for grading.
2. **Make sure you've added `mihail911`, `isaackann`, and `vdaita` as collaborators on your assignment repository.**
3. Submit via Gradescope.
4. **Optional: Clean up when you're done**: Remove the server from your agent config (`claude mcp remove <name>`, or delete the local `.mcp.json` entry), delete your cached token file, and revoke the OAuth app's access from the provider's account settings.
