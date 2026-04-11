import { existsSync, readdirSync, readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEEK4 = path.join(__dirname, '..');
const ROOT = path.join(WEEK4, '..');
const CLAUDE_DIR = path.join(ROOT, '.claude');

// Parse YAML-style frontmatter from a markdown file
function parseFrontmatter(content) {
  const match = content.match(/^---\s*\n([\s\S]*?)\n---/);
  if (!match) return {};
  const fm = {};
  for (const line of match[1].split('\n')) {
    const colonIdx = line.indexOf(':');
    if (colonIdx > -1) {
      const key = line.slice(0, colonIdx).trim();
      const val = line.slice(colonIdx + 1).trim();
      if (key) fm[key] = val;
    }
  }
  return fm;
}

function getSkills() {
  const skillsDir = path.join(CLAUDE_DIR, 'skills');
  if (!existsSync(skillsDir)) return [];
  return readdirSync(skillsDir, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => ({ name: d.name, file: path.join(skillsDir, d.name, 'SKILL.md') }))
    .filter(s => existsSync(s.file));
}

function getAgents() {
  const agentsDir = path.join(CLAUDE_DIR, 'agents');
  if (!existsSync(agentsDir)) return [];
  return readdirSync(agentsDir)
    .filter(f => f.endsWith('.md'))
    .map(f => ({ name: f.replace('.md', ''), file: path.join(agentsDir, f) }));
}

// ── Design (20 pts) ────────────────────────────────────────────────────────

describe('Design', () => {
  test('.claude/skills/ directory exists', () => {
    expect(existsSync(path.join(CLAUDE_DIR, 'skills'))).toBe(true);
  });

  test('.claude/agents/ directory exists', () => {
    expect(existsSync(path.join(CLAUDE_DIR, 'agents'))).toBe(true);
  });

  test('.claude/settings.json exists', () => {
    expect(existsSync(path.join(CLAUDE_DIR, 'settings.json'))).toBe(true);
  });

  test('.claude/settings.json has at least one hook configured', () => {
    const settings = JSON.parse(
      readFileSync(path.join(CLAUDE_DIR, 'settings.json'), 'utf8')
    );
    expect(settings.hooks).toBeDefined();
    expect(Object.keys(settings.hooks).length).toBeGreaterThan(0);
  });
});

// ── Functionality (35 pts) ─────────────────────────────────────────────────

describe('Functionality', () => {
  test('at least 2 skills exist', () => {
    expect(getSkills().length).toBeGreaterThanOrEqual(2);
  });

  test('at least 1 subagent exists', () => {
    expect(getAgents().length).toBeGreaterThanOrEqual(1);
  });

  test('each skill has name in frontmatter', () => {
    const skills = getSkills();
    expect(skills.length).toBeGreaterThan(0);
    for (const skill of skills) {
      const fm = parseFrontmatter(readFileSync(skill.file, 'utf8'));
      expect(fm.name).toBeTruthy();
    }
  });

  test('each skill has allowed-tools in frontmatter', () => {
    const skills = getSkills();
    expect(skills.length).toBeGreaterThan(0);
    for (const skill of skills) {
      const fm = parseFrontmatter(readFileSync(skill.file, 'utf8'));
      expect(fm['allowed-tools']).toBeTruthy();
    }
  });

  test('each subagent has name in frontmatter', () => {
    const agents = getAgents();
    expect(agents.length).toBeGreaterThan(0);
    for (const agent of agents) {
      const fm = parseFrontmatter(readFileSync(agent.file, 'utf8'));
      expect(fm.name).toBeTruthy();
    }
  });

  test('at least one skill references a Jikan MCP tool', () => {
    const jikanTools = ['search_anime', 'get_anime', 'get_top_anime', 'search_manga'];
    const found = getSkills().some(skill => {
      const content = readFileSync(skill.file, 'utf8');
      return jikanTools.some(tool => content.includes(tool));
    });
    expect(found).toBe(true);
  });

  test('at least one subagent references a Jikan MCP tool', () => {
    const jikanTools = ['search_anime', 'get_anime', 'get_top_anime', 'search_manga'];
    const found = getAgents().some(agent => {
      const content = readFileSync(agent.file, 'utf8');
      return jikanTools.some(tool => content.includes(tool));
    });
    expect(found).toBe(true);
  });
});

// ── Documentation (20 pts) ─────────────────────────────────────────────────

describe('Documentation', () => {
  test('week4/CLAUDE.md exists', () => {
    expect(existsSync(path.join(WEEK4, 'CLAUDE.md'))).toBe(true);
  });

  test('week4/CLAUDE.md describes the architecture', () => {
    const content = readFileSync(path.join(WEEK4, 'CLAUDE.md'), 'utf8');
    expect(/skill|agent|hook/i.test(content)).toBe(true);
  });

  test('each skill has a description in frontmatter', () => {
    const skills = getSkills();
    expect(skills.length).toBeGreaterThan(0);
    for (const skill of skills) {
      const fm = parseFrontmatter(readFileSync(skill.file, 'utf8'));
      expect(fm.description).toBeTruthy();
    }
  });

  test('each skill body has instructions (> 50 chars)', () => {
    const skills = getSkills();
    expect(skills.length).toBeGreaterThan(0);
    for (const skill of skills) {
      const content = readFileSync(skill.file, 'utf8');
      const bodyMatch = content.match(/^---[\s\S]*?---\s*\n([\s\S]*)$/);
      const body = bodyMatch?.[1] ?? '';
      expect(body.trim().length).toBeGreaterThan(50);
    }
  });
});

// ── Code Quality (15 pts) ──────────────────────────────────────────────────

describe('Code Quality', () => {
  test('each skill has argument-hint in frontmatter', () => {
    const skills = getSkills();
    expect(skills.length).toBeGreaterThan(0);
    for (const skill of skills) {
      const fm = parseFrontmatter(readFileSync(skill.file, 'utf8'));
      expect(fm['argument-hint']).toBeTruthy();
    }
  });

  test('each subagent has a description in frontmatter', () => {
    const agents = getAgents();
    expect(agents.length).toBeGreaterThan(0);
    for (const agent of agents) {
      const fm = parseFrontmatter(readFileSync(agent.file, 'utf8'));
      expect(fm.description).toBeTruthy();
    }
  });

  test('each subagent has a focused system prompt (body > 100 chars)', () => {
    const agents = getAgents();
    expect(agents.length).toBeGreaterThan(0);
    for (const agent of agents) {
      const content = readFileSync(agent.file, 'utf8');
      const bodyMatch = content.match(/^---[\s\S]*?---\s*\n([\s\S]*)$/);
      const body = bodyMatch?.[1] ?? '';
      expect(body.trim().length).toBeGreaterThan(100);
    }
  });
});
