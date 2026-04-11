---
name: recommend
description: Get personalized anime or manga recommendations via the Jikan MCP server. Triggered when the user asks for recommendations, suggestions, or "what should I watch/read next".
context: fork
allowed-tools: Bash
argument-hint: "[genre or preference]  e.g. action completed, slice of life manga"
---

# Anime / Manga Recommender

Generate personalized anime or manga recommendations using the Jikan MCP server and the `anime-researcher` subagent.

## Steps

1. **Parse preferences from `$ARGUMENTS`**

   Extract any of the following if present:
   - Media type: `anime` (default) or `manga`
   - Genre keywords (e.g. action, romance, horror, sci-fi, slice of life)
   - Status filter: `airing`, `complete`, `upcoming` (anime) / `publishing`, `complete` (manga)
   - Type filter: `tv`, `movie`, `ova` (anime) / `manga`, `novel`, `manhwa` (manga)

   If `$ARGUMENTS` is empty, ask:
   > "What kind of anime or manga are you looking for? (e.g. genre, mood, or type)"

2. **Fetch candidates using the Jikan MCP tools**

   Run 2–3 searches to cover different angles of the user's preference:

   | Pass | Tool | Strategy |
   |------|------|----------|
   | 1 | `search_anime` / `search_manga` | Primary genre keyword, `limit: 10` |
   | 2 | `get_top_anime` | `type` filter if specified, `limit: 10` |
   | 3 | `search_anime` / `search_manga` | Secondary keyword or related genre, `limit: 5` |

3. **Delegate deep research to the `anime-researcher` subagent**

   For the top 3–5 candidates, invoke the `anime-researcher` agent to fetch full details and produce a structured report for each.

4. **Present recommendations**

   Return a ranked list of 3–5 recommendations in this format:

   ```
   ## Recommendations

   ### 1. <Title> (<Year>)
   Type: TV · Episodes: 26 · Score: 8.7 · Rank: #12
   Genres: Action, Adventure
   Why it fits: <1–2 sentences tying it to the user's request>
   Synopsis: <2–3 sentences>
   URL: https://myanimelist.net/anime/...

   ### 2. ...
   ```

## Error Handling

- If no results match: relax the filters (remove `status` or `type`) and try again.
- If the Jikan MCP server is unavailable: tell the user to run `cd week3 && npm install && node server/main.js`.
- If rate-limited (429): wait a moment and retry once.
