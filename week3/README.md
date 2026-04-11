# Week 3 — GitHub Issues MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that wraps the GitHub REST API, letting Claude Desktop, Cursor, or any MCP-compatible client manage GitHub Issues directly from a conversation.

## Features

| Tool | Description |
|------|-------------|
| `get_repo_info` | Repository metadata (stars, forks, open issues, default branch) |
| `list_issues` | List open / closed / all issues (PRs excluded) |
| `create_issue` | Create a new issue — turn action items into trackable tasks |
| `close_issue` | Close an existing issue by number |

## Prerequisites

- Node.js ≥ 18
- A GitHub [Personal Access Token](https://github.com/settings/tokens) with `repo` scope (for private repos) or no scope (for public repos)

## Setup

### 1. Install dependencies

```bash
# From the week3/ directory
npm install
```

### 2. Set your token

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

Or create a `.env` file in `week3/`:
```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

### 3. Run the server (manual test)

```bash
# From week3/ directory
GITHUB_TOKEN=<token> node server/main.js
```

## Claude Desktop Configuration

Add the following to your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": ["server/main.js"],
      "cwd": "/absolute/path/to/week3",
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

Restart Claude Desktop and you should see the GitHub tools appear.

> For Cursor, add the same JSON block to `.cursor/mcp.json` in your project root.

## Tool Reference

### `get_repo_info`

```
Parameters:
  owner  (string, required) — GitHub username or organization
  repo   (string, required) — Repository name

Returns: name, description, stars, forks, open_issues, default_branch, visibility, url
```

Example invocation in Claude:
> "Show me info about the anthropics/anthropic-sdk-python repo"

---

### `list_issues`

```
Parameters:
  owner     (string, required)
  repo      (string, required)
  state     (string, optional) — "open" | "closed" | "all"  [default: "open"]
  per_page  (integer, optional) — max results, 1-100          [default: 30]

Returns: list of { number, title, state, labels, url }
```

Example:
> "List the open issues in my-org/my-repo"

---

### `create_issue`

```
Parameters:
  owner   (string, required)
  repo    (string, required)
  title   (string, required)
  body    (string, optional) — markdown description
  labels  (array of strings, optional) — must already exist in the repo

Returns: { number, title, url, state }
```

Example:
> "Create a GitHub issue titled 'Add search endpoint for notes' in my-org/week4-app"

---

### `close_issue`

```
Parameters:
  owner         (string, required)
  repo          (string, required)
  issue_number  (integer, required)

Returns: { number, title, state, url }
```

Example:
> "Close issue #42 in my-org/my-repo"

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| Invalid token | Returns a clear `401 Unauthorized` message |
| Repo not found | Returns `404 Not Found` |
| Rate limit exceeded | Returns remaining reset time; warns when < 5 requests left |
| Missing required param | Returns `Invalid arguments` with the missing field name |

## Project Structure

```
week3/
├── server/
│   ├── main.js           # MCP STDIO server
│   └── githubClient.js   # GitHub REST API wrapper
├── package.json
└── README.md
```
