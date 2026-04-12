import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

async function createClient() {
  const transport = new StdioClientTransport({
    command: 'node',
    args: [path.join(ROOT, 'server', 'main.js')],
  });
  const client = new Client({ name: 'test', version: '1.0' }, { capabilities: {} });
  await client.connect(transport);
  return { client, transport };
}

// ── Functionality (50 pts) ─────────────────────────────────────────────────

describe('Functionality', () => {
  let client, transport;

  beforeAll(async () => {
    ({ client, transport } = await createClient());
  }, 15000);

  afterAll(async () => {
    await transport.close();
  });

  test('exposes all 4 required tools', async () => {
    const { tools } = await client.listTools();
    const names = tools.map(t => t.name);
    expect(names).toContain('search_anime');
    expect(names).toContain('get_anime');
    expect(names).toContain('get_top_anime');
    expect(names).toContain('search_manga');
  }, 10000);

  test('search_anime returns results with title', async () => {
    const result = await client.callTool({ name: 'search_anime', arguments: { query: 'Naruto' } });
    const text = result.content?.[0]?.text ?? '';
    expect(text.toLowerCase()).toContain('naruto');
  }, 20000);

  test('get_anime returns anime details', async () => {
    // Anime ID 1 is Cowboy Bebop on MAL
    const result = await client.callTool({ name: 'get_anime', arguments: { id: 1 } });
    const text = result.content?.[0]?.text ?? '';
    expect(text.length).toBeGreaterThan(0);
  }, 20000);

  test('get_top_anime returns a list', async () => {
    const result = await client.callTool({ name: 'get_top_anime', arguments: {} });
    const text = result.content?.[0]?.text ?? '';
    expect(text.length).toBeGreaterThan(0);
  }, 20000);

  test('search_manga returns results with title', async () => {
    const result = await client.callTool({ name: 'search_manga', arguments: { query: 'One Piece' } });
    const text = result.content?.[0]?.text ?? '';
    expect(text.toLowerCase()).toContain('one piece');
  }, 20000);
});

// ── Reliability (20 pts) ───────────────────────────────────────────────────

describe('Reliability', () => {
  let client, transport;

  beforeAll(async () => {
    ({ client, transport } = await createClient());
  }, 15000);

  afterAll(async () => {
    await transport.close();
  });

  test('search_anime handles empty query without crashing', async () => {
    await expect(
      client.callTool({ name: 'search_anime', arguments: { query: '' } })
    ).resolves.toBeDefined();
  }, 20000);

  test('get_anime handles non-existent ID without crashing', async () => {
    await expect(
      client.callTool({ name: 'get_anime', arguments: { id: 999999999 } })
    ).resolves.toBeDefined();
  }, 20000);

  test('search_anime handles special characters without crashing', async () => {
    await expect(
      client.callTool({ name: 'search_anime', arguments: { query: '!@#$%^&*()' } })
    ).resolves.toBeDefined();
  }, 20000);

  test('tools return error message instead of throwing on bad input', async () => {
    const result = await client.callTool({ name: 'get_anime', arguments: { id: -1 } });
    // Should return a content response, not throw
    expect(result.content).toBeDefined();
  }, 20000);
});
