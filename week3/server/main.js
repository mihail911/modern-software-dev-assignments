// MCP Server — Jikan API
// 透過 STDIO transport 將 Jikan REST API 包裝成 MCP tools
// 讓 Claude Code / Claude Desktop 可以直接呼叫 anime/manga 相關功能

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import https from 'https';

const JIKAN_BASE = 'https://api.jikan.moe/v4';

// 發送 HTTPS GET 請求並回傳 JSON
// 使用 Node.js 內建的 https 模組（而非 fetch），
// 並強制使用 IPv4（family: 4）避免 IPv6 連線失敗的問題
function httpsGet(url, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { family: 4 }, (res) => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`Jikan API error: ${res.statusCode}`));
        } else {
          try { resolve(JSON.parse(data)); }
          catch (e) { reject(new Error('Failed to parse JSON response')); }
        }
      });
    });
    // 超過 timeoutMs 毫秒沒有回應就中斷請求
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    req.on('error', reject);
  });
}

// 呼叫 Jikan API 的統一入口，path 為 /v4 之後的路徑
async function jikanFetch(path) {
  return httpsGet(`${JIKAN_BASE}${path}`);
}

// 將字串包裝成 MCP 規定的 content 回傳格式
function textContent(text) {
  return { content: [{ type: 'text', text: String(text) }] };
}

// 宣告這個 MCP server 提供的 4 個 tools 及其參數 schema
const TOOLS = [
  {
    name: 'search_anime',
    description: 'Search anime by title using the Jikan API',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Anime title to search for' },
      },
      required: ['query'],
    },
  },
  {
    name: 'get_anime',
    description: 'Get full details for an anime by its MyAnimeList ID',
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'number', description: 'MyAnimeList anime ID' },
      },
      required: ['id'],
    },
  },
  {
    name: 'get_top_anime',
    description: 'Fetch the current top-ranked anime list from MyAnimeList',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'search_manga',
    description: 'Search manga by title using the Jikan API',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Manga title to search for' },
      },
      required: ['query'],
    },
  },
];

// 搜尋動漫，回傳前 10 筆結果（MAL ID、標題、年份、評分）
async function handleSearchAnime(args) {
  const { query } = args;
  if (!query) {
    return textContent('Error: query parameter is required');
  }
  const data = await jikanFetch(`/anime?q=${encodeURIComponent(query)}&limit=10`);
  const results = (data.data ?? []).map(a =>
    `[${a.mal_id}] ${a.title} (${a.aired?.prop?.from?.year ?? 'N/A'}) — Score: ${a.score ?? 'N/A'}`
  );
  return textContent(results.length ? results.join('\n') : 'No results found.');
}

// 依 MAL ID 取得單部動漫的完整資訊
async function handleGetAnime(args) {
  const { id } = args;
  const data = await jikanFetch(`/anime/${id}`);
  const a = data.data;
  const text = [
    `Title: ${a.title}`,
    `English: ${a.title_english ?? 'N/A'}`,
    `MAL ID: ${a.mal_id}`,
    `Type: ${a.type ?? 'N/A'}`,
    `Episodes: ${a.episodes ?? 'N/A'}`,
    `Status: ${a.status ?? 'N/A'}`,
    `Score: ${a.score ?? 'N/A'} (${a.scored_by ?? 0} users)`,
    `Rank: ${a.rank ?? 'N/A'}`,
    `Genres: ${(a.genres ?? []).map(g => g.name).join(', ') || 'N/A'}`,
    `Synopsis: ${a.synopsis ?? 'N/A'}`,
  ].join('\n');
  return textContent(text);
}

// 取得 MAL 目前排名前 25 的動漫清單
async function handleGetTopAnime() {
  const data = await jikanFetch('/top/anime?limit=25');
  const results = (data.data ?? []).map((a, i) =>
    `${i + 1}. [${a.mal_id}] ${a.title} — Score: ${a.score ?? 'N/A'}`
  );
  return textContent(results.length ? results.join('\n') : 'No results found.');
}

// 搜尋漫畫，回傳前 10 筆結果（MAL ID、標題、評分、狀態）
async function handleSearchManga(args) {
  const { query } = args;
  if (!query) {
    return textContent('Error: query parameter is required');
  }
  const data = await jikanFetch(`/manga?q=${encodeURIComponent(query)}&limit=10`);
  const results = (data.data ?? []).map(m =>
    `[${m.mal_id}] ${m.title} — Score: ${m.score ?? 'N/A'}, Status: ${m.status ?? 'N/A'}`
  );
  return textContent(results.length ? results.join('\n') : 'No results found.');
}

// 建立 MCP server 實例，宣告支援 tools 能力
const server = new Server(
  { name: 'jikan-mcp-server', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

// 處理 listTools 請求：回傳所有可用 tools 的定義
server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

// 處理 callTool 請求：依 tool 名稱分派到對應的 handler
// 所有錯誤都 catch 住並以文字回傳，確保 server 不會崩潰
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    switch (name) {
      case 'search_anime':
        return await handleSearchAnime(args ?? {});
      case 'get_anime':
        return await handleGetAnime(args ?? {});
      case 'get_top_anime':
        return await handleGetTopAnime();
      case 'search_manga':
        return await handleSearchManga(args ?? {});
      default:
        return textContent(`Error: Unknown tool "${name}"`);
    }
  } catch (err) {
    return textContent(`Error: ${err.message}`);
  }
});

// 啟動 STDIO transport 並連接 server
// STDIO 模式：parent process 透過 stdin/stdout 與此 server 溝通
const transport = new StdioServerTransport();
await server.connect(transport);
