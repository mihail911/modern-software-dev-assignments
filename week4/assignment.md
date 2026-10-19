# Week 4: Make a Messy Repo Agent-Ready

## Assignment Overview

Pick an unfamiliar open-source repo and write the agent configuration file it's missing: an `AGENTS.md` or a `CLAUDE.md`, your choice. The point is practicing the **structure and content** of a good one: what belongs in it, what doesn't, and how specific it has to be to actually be useful.

### Learning Goals

- **Orient** in an unfamiliar codebase well enough to describe it accurately.
- **Encode** that knowledge as durable, executable repo context.
- **Judge** what belongs in an agent config file and what is noise.

## Materials

- **[AGENTS.md format](https://agents.md/)**
- **[Writing a Good CLAUDE.md](https://www.hlyr.dev/blog/writing-a-good-claude-md)**
- **[Claude Code best practices](https://code.claude.com/docs/en/best-practices)**

## Sample Repositories

| Repo |
|---|
| [bigskysoftware/htmx](https://github.com/bigskysoftware/htmx) |
| [iamkun/dayjs](https://github.com/iamkun/dayjs) |
| [slidevjs/slidev](https://github.com/slidevjs/slidev) |
| [pixijs/pixijs](https://github.com/pixijs/pixijs) |
| [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |
| [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) |
| [vuejs/vue](https://github.com/vuejs/vue) |
| [react/react](https://github.com/react/react) |

You may use your own instead, as long as it's a repo you did **not** write.

## Part I: Explore the Repo (20 pts)

Fork it, then find out how it actually works:

- How is it built, tested, linted, and run? Get the **exact** commands.
- What are the major directories and entry points?
- What conventions does the code follow that nothing states out loud?
- What's generated, vendored, or otherwise off-limits to hand-editing?

Use an agent to help you explore. That's the fastest way to do it.

## Part II: Write the File (55 pts)

Write `AGENTS.md` or `CLAUDE.md` at the root of your fork. Cover at least:

- **What the project is**: a couple of sentences, so the agent knows the domain.
- **Architecture**: key directories, entry points, how the pieces relate.
- **Commands**: build, test, lint, run. Exact and copy-pasteable: `pixi run pytest tests/`, not "run the tests."
- **Conventions**: the testing idiom, style rules, naming patterns the repo actually uses.
- **Guardrails**: what not to touch, and the wrong-but-plausible command or directory worth warning about.

The quality bar: every line should be **specific to this repo**. Generic advice a competent model already follows is padding, and so is restating the README. The file loads into every context, so length is a cost. Keep it around a page.

## Part III: Try It Out (25 pts)

Run two small tasks against the repo with your file in place, something like running the test suite and making a small change.

Report what you saw: did the agent use your commands and follow your conventions? Then name **one thing you'd change** about the file based on watching it get used.

## Deliverables

In `week4/`: your `AGENTS.md` or `CLAUDE.md`, a completed `writeup.md`, and a link to your fork.

## Evaluation Rubric (100 pts total)

| Part | Points | What earns full credit |
|---|---|---|
| I. Exploration | 20 | Accurate account of build/test/lint commands, structure, and conventions |
| II. The file | 55 | All five areas covered, specific to the repo, exact commands, concise |
| III. Trying it | 25 | Two tasks run and reported honestly, plus one grounded improvement |

## SUBMISSION INSTRUCTIONS

1. Make sure you have all changes pushed to your remote repository for grading.
2. **Make sure you've added `mihail911`, `isaackann`, and `vdaita` as collaborators on your assignment repository.**
3. Submit via Gradescope.
