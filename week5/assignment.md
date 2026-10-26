# Week 5: Agent-Readiness Report

## Assignment Overview

Run Factory's agent-readiness report on three open-source repos, record what it found, then explain the pattern. Generating a report is one command per repo. The work is in **reading three of them side by side** and saying something defensible about where repos fall short.

### Learning Goals

- **Assess** unfamiliar repos against a standard readiness framework.
- **Compare** results across repos to separate the general from the specific.
- **Generalize** from three data points to a claim you can defend.

## Materials

- **[Agent Readiness, Factory](https://factory.ai/news/agent-readiness)**: the framework, its eight pillars (Style & Validation, Build System, Testing, Documentation, Dev Environment, Code Quality, Observability, Security & Governance), and its five maturity levels.
- **[Droid](https://factory.ai/)**: Factory's CLI, which is how you generate the report.

## Repositories

Pick **three** from the week 4 list, or your own. Choose repos that differ in language, size, and age, or your analysis in Part II will have nothing to work with.

## Setup

**Prerequisite: Droid CLI.** See the [Droid quickstart guide](https://docs.factory.ai/cli/getting-started/quickstart) for the install command.

## Part I: Run the Reports (40 pts)

Clone each repo, then run `/readiness-report` in Droid from the repo root. It scores 60+ criteria across the eight pillars automatically and returns a maturity level, a per-criterion pass/fail breakdown, and remediation suggestions.

Record for each repo:

- Maturity level, and which criteria are blocking the next one.
- Per-pillar scores.
- The failed criteria, grouped by pillar.
- The report's own top remediation suggestions.

Include the raw output too, but the point is the summary: three reports in a comparable format is what makes Part II possible.

## Part II: Where Repos Fall Short (60 pts)

The analysis. Answer from your data:

- **Which pillars failed most**, and were the failures consistent across all three repos or specific to one?
- **What explains the pattern?** Test the obvious suspects (age, language, size, single-maintainer vs. corporate-backed) and say what your three repos can and cannot support.
- **The cheapest high-leverage fix.** For each repo, the one change that would most improve an agent's odds, and why that one over the report's own top suggestion.
- **Where the framework falls short.** A pillar that mattered less than its weight suggests, a criterion that doesn't fit the kind of repo you scored, or something it misses that clearly affects agents.

Three repos is a small sample. Scope your claims to what it supports.

## Deliverables

In `week5/`: a completed `writeup.md` with your recorded results and analysis, including the raw report output for each repo.

## Evaluation Rubric (100 pts total)

| Part | Points | What earns full credit |
|---|---|---|
| I. Reports | 40 | Three repos scored, results recorded completely and in a comparable format |
| II. Analysis | 60 | Pattern identified and explained from the data; per-repo fix justified; framework critiqued; claims scoped to the sample |

## SUBMISSION INSTRUCTIONS

1. Make sure you have all changes pushed to your remote repository for grading.
2. **Make sure you've added `mihail911`, `isaackann`, and `vdaita` as collaborators on your assignment repository.**
3. Submit via Gradescope.
