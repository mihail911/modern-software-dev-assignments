import { readFileSync, writeFileSync } from 'fs';

const results = JSON.parse(readFileSync('results.json', 'utf8'));

// Rubric: Functionality(35) + Design(20) + Documentation(20) + Code Quality(15) = 90
const RUBRIC = {
  'Design': {
    points: 20,
    tests: [
      '.claude/skills/ directory exists',
      '.claude/agents/ directory exists',
      '.claude/settings.json exists',
      '.claude/settings.json has at least one hook configured',
    ],
  },
  'Functionality': {
    points: 35,
    tests: [
      'at least 2 skills exist',
      'at least 1 subagent exists',
      'each skill has name in frontmatter',
      'each skill has allowed-tools in frontmatter',
      'each subagent has name in frontmatter',
      'at least one skill references a Jikan MCP tool',
      'at least one subagent references a Jikan MCP tool',
    ],
  },
  'Documentation': {
    points: 20,
    tests: [
      'week4/CLAUDE.md exists',
      'week4/CLAUDE.md describes the architecture',
      'each skill has a description in frontmatter',
      'each skill body has instructions (> 50 chars)',
    ],
  },
  'Code Quality': {
    points: 15,
    tests: [
      'each skill has argument-hint in frontmatter',
      'each subagent has a description in frontmatter',
      'each subagent has a focused system prompt (body > 100 chars)',
    ],
  },
};

const passedTests = new Set(
  results.testResults
    .flatMap(f => f.testResults)
    .filter(t => t.status === 'passed')
    .map(t => t.title)
);

let totalScore = 0;
let report = '## Week 4 Autograder Results\n\n';

for (const [category, { points, tests }] of Object.entries(RUBRIC)) {
  const passed = tests.filter(t => passedTests.has(t)).length;
  const earned = Math.round((passed / tests.length) * points);
  totalScore += earned;

  report += `### ${category} — ${earned}/${points}\n`;
  for (const t of tests) {
    report += `- ${passedTests.has(t) ? '✅' : '❌'} ${t}\n`;
  }
  report += '\n';
}

report += `---\n**Total: ${totalScore} / 90**\n`;

writeFileSync('score.txt', report);
console.log(report);
