import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { JikanClient } from "./jikanClient.js";

const client = new JikanClient();

const server = new McpServer({
  name: "jikan-mcp-server",
  version: "1.0.0",
});

// Tool: search_anime
server.tool(
  "search_anime",
  "Search for anime on MyAnimeList by keyword",
  {
    q: z.string().describe("Search query (anime title or keywords)"),
    limit: z.number().int().min(1).max(25).default(10).describe("Number of results to return (1–25)"),
    type: z
      .enum(["tv", "movie", "ova", "special", "ona", "music"])
      .optional()
      .describe("Filter by anime type"),
    status: z
      .enum(["airing", "complete", "upcoming"])
      .optional()
      .describe("Filter by airing status"),
    rating: z
      .enum(["g", "pg", "pg13", "r17", "r", "rx"])
      .optional()
      .describe("Filter by age rating (g, pg, pg13, r17, r, rx)"),
  },
  async ({ q, limit, type, status, rating }) => {
    const results = await client.searchAnime(q, { limit, type, status, rating });
    return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
  }
);

// Tool: get_anime
server.tool(
  "get_anime",
  "Get full details for an anime by its MyAnimeList ID",
  {
    id: z.number().int().positive().describe("MyAnimeList anime ID (mal_id)"),
  },
  async ({ id }) => {
    const anime = await client.getAnime(id);
    return { content: [{ type: "text", text: JSON.stringify(anime, null, 2) }] };
  }
);

// Tool: get_top_anime
server.tool(
  "get_top_anime",
  "Get the top-ranked anime on MyAnimeList",
  {
    limit: z.number().int().min(1).max(25).default(10).describe("Number of results to return (1–25)"),
    type: z
      .enum(["tv", "movie", "ova", "special", "ona", "music"])
      .optional()
      .describe("Filter by anime type"),
  },
  async ({ limit, type }) => {
    const results = await client.getTopAnime({ limit, type });
    return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
  }
);

// Tool: search_manga
server.tool(
  "search_manga",
  "Search for manga on MyAnimeList by keyword",
  {
    q: z.string().describe("Search query (manga title or keywords)"),
    limit: z.number().int().min(1).max(25).default(10).describe("Number of results to return (1–25)"),
    type: z
      .enum(["manga", "novel", "lightnovel", "oneshot", "doujin", "manhwa", "manhua"])
      .optional()
      .describe("Filter by manga type"),
    status: z
      .enum(["publishing", "complete", "hiatus", "discontinued", "upcoming"])
      .optional()
      .describe("Filter by publication status"),
  },
  async ({ q, limit, type, status }) => {
    const results = await client.searchManga(q, { limit, type, status });
    return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[jikan-mcp-server] Server running on STDIO");
