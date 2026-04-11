# Week 3 — Build a Custom MCP Server

Design and implement a Model Context Protocol (MCP) server that wraps a real external API.

## Tasks

1. Choose an external API and document which endpoints you'll use.
2. Expose at least **two MCP tools** with typed parameters.
3. Implement basic resilience:
   - Graceful errors for HTTP failures, timeouts, and empty results.
   - Respect API rate limits (simple backoff or user-facing warning).
4. Provide clear setup instructions, environment variables, and run commands.
5. Choose one deployment mode:
   - **Local**: STDIO server, runnable from your machine.
   - **Remote**: HTTP server accessible over the network. *(extra credit)*
6. *(Optional)* Add authentication — API key or OAuth2 bearer tokens. *(extra credit)*

## Evaluation Rubric (90 pts)

| Category | Points | Criteria |
|----------|--------|----------|
| Functionality | 35 | 2+ tools implemented, correct API integration, meaningful outputs |
| Reliability | 20 | Input validation, error handling, logging, rate-limit awareness |
| Developer Experience | 20 | Clear setup/docs, easy to run locally, sensible folder structure |
| Code Quality | 15 | Readable code, descriptive names, minimal complexity |
| Extra Credit | +10 | +5 remote HTTP server · +5 auth (API key or OAuth2) |
