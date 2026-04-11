---
name: anime
description: Search and explore anime or manga via the Jikan (MyAnimeList) MCP server. Triggered when the user wants to find anime, get anime details, see top anime, or search for manga.
context: fork
allowed-tools: Bash
argument-hint: "[query]  e.g. cowboy bebop"
---

# Anime / Manga Explorer

Search and explore anime and manga using the Jikan MCP server.

## Steps

1. **Determine intent from `$ARGUMENTS`**

   - If `$ARGUMENTS` looks like an integer → call `get_anime` with that ID
   - If `$ARGUMENTS` starts with `top` → call `get_top_anime`
   - If `$ARGUMENTS` starts with `manga` → call `search_manga` with the remaining text as query
   - Otherwise → call `search_anime` with `$ARGUMENTS` as the query

   If `$ARGUMENTS` is empty, ask the user:
   > "What would you like to look up? (e.g. a title, 'top', or 'manga <title>')"

2. **Call the appropriate Jikan MCP tool**

   | Intent | Tool | Key params |
   |--------|------|------------|
   | Search anime | `search_anime` | `q`, `limit: 10` |
   | Anime by ID | `get_anime` | `id` |
   | Top anime | `get_top_anime` | `limit: 10` |
   | Search manga | `search_manga` | `q`, `limit: 10` |

3. **Present results**

   For lists, print a numbered table with columns: **#**, **Title**, **Type**, **Score**, **Status**, **URL**

   For a single anime (`get_anime`), print a detail card:
   ```
   Title:    ...
   English:  ...
   Type:     ...   Episodes: ...
   Status:   ...   Score: ...  Rank: #...
   Genres:   ...
   Studios:  ...
   Aired:    ...
   Synopsis: ... (first 200 chars)
   URL:      ...
   ```

## Error Handling

- If the Jikan MCP server is unavailable: tell the user to run `cd week3 && npm install && node server/main.js` and configure it in Claude Code MCP settings.
- If a 404 is returned: tell the user the ID or title was not found.
- If a rate-limit error (429) is returned: wait a moment and retry once, then report the error if it persists.
