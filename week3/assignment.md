# Week 3: Author a Skill

## Assignment Overview

A skill is where a repeatable procedure lives: the checklist you would otherwise paste into chat every time. This week you will author one, point it at a real open-source repo, and find out whether it fires when it should and holds up when it runs.

### Learning Goals

- **Identify** a workflow worth encoding rather than re-explaining.
- **Author** a skill whose description triggers reliably and whose body encodes judgment, not just steps.
- **Validate** that it fires when it should and holds up when it runs.

## Materials

- **[agentskills.io](https://agentskills.io)**: the open standard, which most agents now support.
- **[Claude Code skills docs](https://code.claude.com/docs/en/skills)**: one implementation, useful for file layout and frontmatter.
- **[Sample skills](https://github.com/mattpocock/skills)**: read a few before writing yours.

## Repositories

Pick one to target, or use your own.

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

## Part I: Pick the Workflow (15 pts)

Choose something you would run **more than once**, with at least one real decision point in it. A single fixed command is a shell alias, not a skill.

Ideas, if you'd rather not invent one:

| Skill | What it does |
|---|---|
| Issue to implementation plan | Read an issue, inspect the relevant code, surface ambiguities, produce a step-by-step plan |
| PR review | Review a diff against a fixed rubric, citing exact files and lines, separating blockers from suggestions |
| Bug reproduction | Build the smallest reproduction, form hypotheses, run targeted tests, and only then propose a fix |
| Test writer | Decide what behaviors need tests, match the repo's conventions, write them, run them |
| Repo onboarding | Given a task like "add authentication," orient a new developer: architecture, commands, likely change points |
| Codebase migration | Encode detection rules and safe transformation steps for something like Pydantic v1 to v2 |
| Dependency upgrade | Read the changelog, identify breaking changes, apply them, verify |
| Screenshot to implementation | Implement a UI from a screenshot using the repo's existing components, then verify visually |
| Performance investigation | Establish a baseline, profile before changing anything, make one targeted fix, compare |
| Security review | Inspect new code for one focused category, requiring evidence rather than speculative warnings |

Run it manually once first. You cannot encode a procedure you have not done.

## Part II: Author the Skill (60 pts)

Write a `SKILL.md` following the Agent Skills standard, placed wherever your agent discovers skills. Claude Code, Cursor, and others all read the same format; check your agent's docs for the directory it looks in. What earns credit:

- **A description that triggers.** It has to say what the skill does *and when to use it*, in the words someone would actually type. Most skills fail by never firing.
- **Judgment, not just steps.** Say how to decide when the situation is ambiguous, and what not to do. A restatement of the tool documentation teaches nothing.
- **Exact commands** for the deterministic parts.
- **A lean body.** The body loads into context whenever the skill fires, so push long reference material into supporting files in the skill directory and reference them, rather than inlining everything.

## Part III: Test It (25 pts)

- **Triggering**: three prompts that should fire it, and one near-miss that should *not*. Report what actually happened. If the near-miss fired it, your description is too broad.
- **Running**: use it end to end on your chosen repo and show the result.

## Deliverables

In `week3/`: your skill directory (`SKILL.md` plus any supporting files) and a completed `writeup.md`.

## Evaluation Rubric (100 pts total)

| Part | Points | What earns full credit |
|---|---|---|
| I. The workflow | 15 | Genuinely repeatable, has a real decision point, run manually first |
| II. The skill | 60 | Description that triggers, encoded judgment, exact commands, lean body with supporting files where warranted |
| III. Testing | 25 | Trigger tests including a near-miss, plus an end-to-end run |

## SUBMISSION INSTRUCTIONS

1. Make sure you have all changes pushed to your remote repository for grading.
2. **Make sure you've added `mihail911`, `isaackann`, and `vdaita` as collaborators on your assignment repository.**
3. Submit via Gradescope.
