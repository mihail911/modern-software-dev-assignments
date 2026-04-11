# Week 3 — Build a Custom MCP Server

Design and implement a Model Context Protocol (MCP) server that wraps a real external API. You may:
- Run it **locally** (STDIO transport) and integrate with an MCP client (like Claude Desktop).
- Or run it **remotely** (HTTP transport) and call it from a model agent or client. This is harder but earns extra credit.

Bonus points for adding authentication (API keys or OAuth2) aligned with the MCP Authorization spec.

## Learning goals
- Understand core MCP capabilities: tools, resources, prompts.
- Implement tool definitions with typed parameters and robust error handling.
- Follow logging and transport best practices (no stdout for STDIO servers).
- Optionally implement authorization flows for HTTP transports.

## Requirements
1. Choose an external API and document which endpoints you’ll use. Examples: weather, GitHub issues, Notion pages, movie/TV databases, calendar, task managers, finance/crypto, travel, sports stats.
2. Expose at least two MCP tools
3. Implement basic resilience:
   - Graceful errors for HTTP failures, timeouts, and empty results.
   - Respect API rate limits (e.g., simple backoff or user-facing warning).
4. Packaging and docs:
   - Provide clear setup instructions, environment variables, and run commands.
   - Include an example invocation flow (what to type/click in the client to trigger the tools).
5. Choose one deployment mode:
   - Local: STDIO server, runnable from your machine and discoverable by Claude Desktop or an AI IDE like Cursor.
   - Remote: HTTP server accessible over the network, callable by an MCP-aware client or an agent runtime. Extra credit if deployed and reachable.
6. (Optional) Bonus: Authentication
   - API key support via environment variable and client configuration; or
   - OAuth2-style bearer tokens for HTTP transport, validating token audience and never passing tokens through to upstream APIs.

## Deliverables
- Source code under `week3/` (suggested: `week3/server/` with a clear entrypoint like `main.py` or `app.py`).
- `week3/README.md` with:
  - Prerequisites, environment setup, and run instructions (local and/or remote).
  - How to configure the MCP client (Claude Desktop example for local) or agent runtime for remote.
  - Tool reference: names, parameters, example inputs/outputs, and expected behaviors.

## Evaluation rubric (90 pts total)
- Functionality (35): Implements 2+ tools, correct API integration, meaningful outputs.
- Reliability (20): Input validation, error handling, logging, rate-limit awareness.
- Developer experience (20): Clear setup/docs, easy to run locally; sensible folder structure.
- Code quality (15): Readable code, descriptive names, minimal complexity, type hints where applicable.
- Extra credit (10):
  - +5 Remote HTTP MCP server, callable by an agent/client such as the OpenAI/Claude SDK.
  - +5 Auth implemented correctly (API key or OAuth2 with audience validation).

## Helpful references
- [MCP Server Quickstart](https://modelcontextprotocol.io/quickstart/server) *(you may not submit this exact example)*
- [MCP Authorization (HTTP)](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
- [Remote MCP on Cloudflare](https://developers.cloudflare.com/agents/guides/remote-mcp-server/) — use the MCP inspector to debug locally before deploying
- [Deploy MCP Servers to Vercel](https://vercel.com/docs/mcp/deploy-mcp-servers-to-vercel) — free tier available