import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { GitHubClient } from "./githubClient.js";

const client = new GitHubClient(process.env.GITHUB_TOKEN);

const server = new McpServer({
  name: "github-mcp-server",
  version: "1.0.0",
});

// Tool: get_repo_info
server.tool(
  "get_repo_info",
  "Fetch repository metadata (stars, forks, open issues, default branch)",
  {
    owner: z.string().describe("GitHub username or organization"),
    repo: z.string().describe("Repository name"),
  },
  async ({ owner, repo }) => {
    const info = await client.getRepoInfo(owner, repo);
    return {
      content: [{ type: "text", text: JSON.stringify(info, null, 2) }],
    };
  }
);

// Tool: list_issues
server.tool(
  "list_issues",
  "List issues in a repository (PRs excluded)",
  {
    owner: z.string().describe("GitHub username or organization"),
    repo: z.string().describe("Repository name"),
    state: z
      .enum(["open", "closed", "all"])
      .default("open")
      .describe('Issue state filter: "open", "closed", or "all"'),
    per_page: z
      .number()
      .int()
      .min(1)
      .max(100)
      .default(30)
      .describe("Maximum number of results (1–100)"),
  },
  async ({ owner, repo, state, per_page }) => {
    const issues = await client.listIssues(owner, repo, state, per_page);
    return {
      content: [{ type: "text", text: JSON.stringify(issues, null, 2) }],
    };
  }
);

// Tool: create_issue
server.tool(
  "create_issue",
  "Create a new issue in a repository",
  {
    owner: z.string().describe("GitHub username or organization"),
    repo: z.string().describe("Repository name"),
    title: z.string().describe("Issue title"),
    body: z.string().optional().describe("Issue body (markdown supported)"),
    labels: z
      .array(z.string())
      .optional()
      .describe("Labels to apply (must already exist in the repo)"),
  },
  async ({ owner, repo, title, body, labels }) => {
    const issue = await client.createIssue(owner, repo, title, body, labels);
    return {
      content: [{ type: "text", text: JSON.stringify(issue, null, 2) }],
    };
  }
);

// Tool: close_issue
server.tool(
  "close_issue",
  "Close an existing issue by number",
  {
    owner: z.string().describe("GitHub username or organization"),
    repo: z.string().describe("Repository name"),
    issue_number: z.number().int().positive().describe("Issue number to close"),
  },
  async ({ owner, repo, issue_number }) => {
    const issue = await client.closeIssue(owner, repo, issue_number);
    return {
      content: [{ type: "text", text: JSON.stringify(issue, null, 2) }],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[github-mcp-server] Server running on STDIO");
