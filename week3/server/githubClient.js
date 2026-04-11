const GITHUB_API = "https://api.github.com";

export class GitHubClient {
  #token;

  constructor(token) {
    if (!token) {
      throw new Error("GITHUB_TOKEN environment variable is required");
    }
    this.#token = token;
  }

  async #request(path, options = {}) {
    const url = `${GITHUB_API}${path}`;
    const res = await fetch(url, {
      ...options,
      headers: {
        Authorization: `Bearer ${this.#token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        ...options.headers,
      },
    });

    // Rate limit warning
    const remaining = Number(res.headers.get("X-RateLimit-Remaining"));
    if (!isNaN(remaining) && remaining < 5) {
      const reset = res.headers.get("X-RateLimit-Reset");
      const resetTime = reset ? new Date(Number(reset) * 1000).toISOString() : "unknown";
      console.error(`[warn] GitHub rate limit low: ${remaining} requests remaining, resets at ${resetTime}`);
    }

    if (res.status === 401) {
      throw new Error("GitHub API: 401 Unauthorized — check your GITHUB_TOKEN");
    }
    if (res.status === 404) {
      throw new Error(`GitHub API: 404 Not Found — ${path}`);
    }
    if (res.status === 403 && remaining === 0) {
      const reset = res.headers.get("X-RateLimit-Reset");
      const resetTime = reset ? new Date(Number(reset) * 1000).toISOString() : "unknown";
      throw new Error(`GitHub API: rate limit exceeded, resets at ${resetTime}`);
    }
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`GitHub API: ${res.status} ${res.statusText} — ${body}`);
    }

    return res.json();
  }

  async getRepoInfo(owner, repo) {
    const data = await this.#request(`/repos/${owner}/${repo}`);
    return {
      name: data.full_name,
      description: data.description ?? "",
      stars: data.stargazers_count,
      forks: data.forks_count,
      open_issues: data.open_issues_count,
      default_branch: data.default_branch,
      visibility: data.visibility,
      url: data.html_url,
    };
  }

  async listIssues(owner, repo, state = "open", per_page = 30) {
    const params = new URLSearchParams({ state, per_page: String(per_page), pulls: "false" });
    const data = await this.#request(`/repos/${owner}/${repo}/issues?${params}`);
    // GitHub issues endpoint also returns PRs; filter them out
    return data
      .filter((issue) => !issue.pull_request)
      .map((issue) => ({
        number: issue.number,
        title: issue.title,
        state: issue.state,
        labels: issue.labels.map((l) => l.name),
        url: issue.html_url,
      }));
  }

  async createIssue(owner, repo, title, body, labels) {
    const payload = { title };
    if (body) payload.body = body;
    if (labels?.length) payload.labels = labels;

    const data = await this.#request(`/repos/${owner}/${repo}/issues`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    return {
      number: data.number,
      title: data.title,
      url: data.html_url,
      state: data.state,
    };
  }

  async closeIssue(owner, repo, issue_number) {
    const data = await this.#request(`/repos/${owner}/${repo}/issues/${issue_number}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: "closed" }),
    });

    return {
      number: data.number,
      title: data.title,
      state: data.state,
      url: data.html_url,
    };
  }
}
