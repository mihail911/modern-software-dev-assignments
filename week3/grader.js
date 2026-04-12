import { readFileSync, writeFileSync } from 'fs';

const results = JSON.parse(readFileSync('results.json', 'utf8'));

// Rubric: Functionality(50) + Reliability(30) + Code Quality(20) = 100
const RUBRIC = {
  'Functionality': {
    points: 50,
    tests: [
      'exposes all 4 required tools',
      'search_anime returns results with title',
      'get_anime returns anime details',
      'get_top_anime returns a list',
      'search_manga returns results with title',
    ],
  },
  'Reliability': {
    points: 30,
    tests: [
      'search_anime handles empty query without crashing',
      'get_anime handles non-existent ID without crashing',
      'search_anime handles special characters without crashing',
      'tools return error message instead of throwing on bad input',
    ],
  },
  'Code Quality': {
    points: 20,
    tests: [],
  },
};

// Collect all passed test titles
const passedTests = new Set(
  results.testResults
    .flatMap(f => f.testResults)
    .filter(t => t.status === 'passed')
    .map(t => t.title)
);

const AUTO_TESTABLE_POINTS = 80; // 50 + 30
let autoScore = 0;
let report = '## Week 3 Autograder Results\n\n';

for (const [category, { points, tests }] of Object.entries(RUBRIC)) {
  if (category === 'Code Quality') continue; // handled separately below
  const passed = tests.filter(t => passedTests.has(t)).length;
  const earned = Math.round((passed / tests.length) * points);
  autoScore += earned;

  report += `### ${category} — ${earned}/${points}\n`;
  for (const t of tests) {
    report += `- ${passedTests.has(t) ? '✅' : '❌'} ${t}\n`;
  }
  report += '\n';
}

// Code Quality: estimated proportionally from the auto-testable score
const CODE_QUALITY_POINTS = 20;
const codeQualityEarned = Math.round((autoScore / AUTO_TESTABLE_POINTS) * CODE_QUALITY_POINTS);
const totalScore = autoScore + codeQualityEarned;

report += `### Code Quality — ${codeQualityEarned}/${CODE_QUALITY_POINTS}\n`;
report += `_Estimated proportionally from automated test results_\n\n`;
report += `---\n**Total: ${totalScore} / 100**\n`;

writeFileSync('score.txt', report);
console.log(report);
