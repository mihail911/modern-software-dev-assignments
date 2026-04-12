import { readFileSync, writeFileSync } from 'fs';

const results = JSON.parse(readFileSync('results.json', 'utf8'));

// Rubric: Functionality(75) + Code Quality(25) = 100
const RUBRIC = {
  'Functionality': {
    points: 75,
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
  'Code Quality': {
    points: 25,
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

report += `---\n**Total: ${totalScore} / 100**\n`;

writeFileSync('score.txt', report);
console.log(report);
