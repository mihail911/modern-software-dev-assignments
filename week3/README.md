# Week 3 — Jikan MCP Server

A Model Context Protocol (MCP) server that wraps the [Jikan REST API](https://jikan.moe/) (unofficial MyAnimeList API), exposing anime and manga data as MCP tools.

## Setup

Requires **Node.js ≥ 18**. No API key needed — Jikan is a public API.

```bash
npm install
node server/main.js
```

## Tools

| Tool | Description |
|------|-------------|
| `search_anime` | Search anime by title |
| `get_anime` | Get full details for an anime by ID |
| `get_top_anime` | Fetch top-ranked anime |
| `search_manga` | Search manga by title |

## Transport

STDIO — compatible with Claude Code and Claude Desktop MCP config.

---

## Bonus — Deploy to Vercel (not graded)

You can expose this MCP server over HTTP using Vercel's serverless functions, making it accessible remotely instead of running it locally via STDIO.

### 1. Add an HTTP transport entry point

Create `api/mcp.js` at the repo root:

```js
import { createServer } from '../server/main.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';

export default async function handler(req, res) {
  const server = createServer();
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
}
```

> This requires refactoring `server/main.js` to export a `createServer()` function instead of calling `server.connect()` directly.

### 2. Add a `vercel.json`

```json
{
  "rewrites": [{ "source": "/mcp", "destination": "/api/mcp" }]
}
```

### 3. Deploy

```bash
npm i -g vercel
vercel
```

### 4. Configure in Claude Code

Replace the STDIO config with the deployed URL:

```json
{
  "mcpServers": {
    "jikan": {
      "type": "http",
      "url": "https://<your-project>.vercel.app/mcp"
    }
  }
}
```
