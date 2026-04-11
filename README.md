# Claude Code Assignments

## Repo Setup

Requires **Node.js ≥ 18**.

```bash
cd week3 && npm install
cd ../week4 && npm install
```

## Developer Automation Layer (Week 3 + Week 4)

Weeks 3 and 4 together build a unified **Developer Automation Layer** powered by the Jikan (MyAnimeList) API.

| Directory | Role |
|-----------|------|
| `week3/` | Jikan MCP Server — wraps the Jikan REST API, exposes 4 tools |
| `week4/` | Claude Code Agent — skills, subagents, and hooks built on top of the MCP server |

## Running Tests

**Week 3** (MCP server — makes live Jikan API calls):

```bash
cd week3 && npm test
```

**Week 4** (agent structure — checks files and frontmatter):

```bash
cd week4 && npm test
```

## Autograding

Grades are calculated automatically when you open a Pull Request to `master`:

1. Push your work to a branch and open a PR targeting `master`
2. GitHub Actions runs the test suite for each week
3. A comment is posted on the PR with the breakdown and total score (out of 90)

Each week is graded independently — week3 and week4 each produce a separate score comment.

## Acknowledgements
[themodernsoftware.dev](https://themodernsoftware.dev/)
