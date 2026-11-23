# Week 6: Agent Cost-Optimization

## Assignment Overview

In this assignment, you will tune a coding agent setup to optimize it for resolving incidents in a production repository. Your optimization goals are:

- Solving incidents **correctly**
- Solving incidents **as cheaply as possible**

Only **Claude Code** and **Codex** harnesses are supported for this assignment. Additionally, to support cost optimization, only the models listed in the assignment **rate card** (see below) can be used with these harnesses.

After optimizing your setup, you will submit it to an autograder that runs it on a held-out incident (for the same repository). Your setup output will be scored for correctness and priced (by token) for cost.

> Note: your setup must be able to run independently (i.e. in headless mode), without user input (there will be none in the autograder)!

The production repository for this assignment is Switchyard, a webhook delivery system.

### Learning Goals

- **Measure** what an agent run costs and where the tokens go.
- **Trade** capability for cost on purpose: model choice, effort, context, decomposition.
- **Build** a setup that works unattended on a task it was not tuned for.

### Trusted Sources

For incidents in Switchyard, the root `task.md` defines the requested outcomes and `app/contracts/` defines the system's required behavior. You may need to resolve inconsistencies in other repository materials against those requirements to optimize your setup.

The incident's `task.md` may identify diagnoses or suggestions; these are leads to investigate. You can check them against the contracts, code, and incident evidence.

## Materials

- **[Claude Code headless mode](https://code.claude.com/docs/en/headless)**, **[Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)**: how each supported harness runs without a terminal.
- **`rate_card.json`**: the price of every allowed model. Frozen for the assignment.

## What You Get

| Path                | What it is                                                                                                                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/`              | Switchyard release 2.3.1: `api` (TypeScript), `router` (Python) and `worker` (Go), a simulated merchant, and everything that has piled up around them. Written for this course, so no model has seen it. |
| `incident/`         | Production logs for one region, collected for the starter ticket.                                                                                                                                        |
| `task.md`           | The starter ticket.                                                                                                                                                                                      |
| `run.sh`            | The entry point, and the baseline: one Claude Code call on the medium model.                                                                                                                             |
| `rate_card.json`    | Allowed models and their prices.                                                                                                                                                                         |
| `tools/practice.py` | Runs your setup the way the grader does and prints scores and cost.                                                                                                                                      |
| `tools/starter/`    | What the starter is scored on: behaviour scenarios and the true impact report. Your agent never sees them: `practice.py` keeps them out of its workspace.                                                |
| `tools/gateway.py`  | The metering gateway the grader uses. `practice.py` starts it for you.                                                                                                                                   |
| `writeup.md`        | The write-up template.                                                                                                                                                                                   |

You need Python 3, Node 24 or newer, and Go to run the services and `practice.py`.

## How a Graded Run Works

The grader copies your `week6/` directory into a fresh container and runs:

```bash
./run.sh task.md
```

from that directory, as an unprivileged user with an empty home directory, no terminal, and a **10-minute limit**.

Before each run the grader **replaces `app/`, `incident/` and `task.md` with its own copies**, and leaves out `tools/starter/` and `out/`. Everything else in `week6/` is your setup and is used exactly as you submitted it: `run.sh`, prompts, a root `AGENTS.md` or `CLAUDE.md`, skills, hooks, subagent definitions, helper scripts.

### Where your setup lives

Edits committed inside `app/` will not apply to graded runs, because the grader puts its own `app/` in place before `run.sh` starts. However, `run.sh` may still rewrite, replace or delete any file (including in `app/`) there before the agent starts.

### What is scored

When `run.sh` exits, two things are taken from the workspace:

- **The repair.** `app/services/api`, `app/services/router` and `app/services/worker` are copied into a clean copy of the release and run through hidden tests.
- **The impact report.** `out/impact.json`, compared with what really happened in `incident/`.

| Score  | How it is computed                                                                   |
| ------ | ------------------------------------------------------------------------------------ |
| repair | share of the incident scenarios passed, times the share of the others passed         |
| impact | 0.6 × F1 on the affected set + 0.4 × the share of the report's figures exactly right |

A run **passes** when both are 1.00. The held-out incident is scored the same way, with its own scenarios and evidence.

### Harnesses and models

Two harnesses are installed in the grader, at pinned versions: **Claude Code** and **Codex**. You may use either, or both in the same `run.sh`.

Every model call goes through the course gateway, which accepts only the models on the rate card:

| Model               | Tier   | Input | Output | Cache read | Cache write |
| ------------------- | ------ | ----- | ------ | ---------- | ----------- |
| `claude-sonnet-5-5` | medium | $2.00 | $10.00 | $0.10      | $2.50       |
| `claude-haiku-5-5`  | small  | $0.10 | $0.50  | $0.01      | $0.125      |
| `gpt-6.1-terra`     | medium | $1.00 | $5.00  | $0.05      | $1.25       |
| `gpt-6-luna`        | small  | $0.10 | $0.50  | $0.01      | $0.125      |

Prices are per million tokens. A request for any other model is refused, and the refusal shows up in your results.

### What `run.sh` can rely on

| Harness     | What `run.sh` can rely on                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Claude Code | `claude -p ... --model claude-haiku-5-5 --permission-mode bypassPermissions` works as is. Background calls go to `claude-haiku-5-5` and are metered like any other. |
| Codex       | `codex exec -m gpt-6-luna --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check ...` works as is.                                       |

Both harnesses are already pointed at the course gateway. When graded, the gateway uses course API keys and your harness holds only a placeholder, so do not put a key of your own in your submission. In practice, your own login or an `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` works (see Part I). Don't write a `run.sh` that depends on which one is in use.

Not available in a graded run: any model off the rate card, self-hosted models, your own accounts and logins, and anything that needs you to answer a prompt.

## How Cost Is Computed

Cost is **tokens times the rate card**, summed over every request the gateway forwarded during the run: input, output, cache reads, and cache writes, each at its own price. Dollar figures reported by a harness are ignored, so a model costs the same whichever harness calls it. Each run starts cold: a new container user, an empty home directory, and no cache from a previous run.

Each run also has a spending cap of **$2.00**. Past it, the gateway refuses further requests.

## Part I: Measure the Baseline (10 pts of the write-up)

Run the baseline and record what it does:

```bash
python tools/practice.py --keep
```

This copies `week6/` to a temporary workspace (without `tools/starter/`), runs `./run.sh task.md` there through the gateway, scores the result the way the grader does, and prints the scores by category, cost, and tokens by model. Logs land in `runs/`: `agent.log` is what `run.sh` printed, and `--keep` also saves the agent's changes to `app/` as `app.diff` and its `out/` directory. Your own `app/` is never modified; keep it clean, because practice scores inside a copy of it and the grader starts from its own.

Practice works with your own login for either harness: your Stanford Claude login for Claude Code, and your ChatGPT login (`codex login`) for Codex. An `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` also works. Practice runs are on your own account, and the gateway meters them against the rate card either way.

Before changing anything, read the logs. Where did the tokens go, which scenarios did the agent miss, and what did it believe that led it there?

## Part II: Tune the Setup (20 pts of the write-up)

Get the agent to resolve incidents the way this system needs them resolved, and bring the cost down. Anything outside the replaced paths is fair game:

| Lever            | Examples                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| Model            | Small instead of medium; a different vendor                                                            |
| Routing          | Medium model diagnoses, small model edits; escalate only when checks fail                              |
| Reasoning effort | Lower effort for mechanical steps                                                                      |
| Context          | A root `AGENTS.md` / `CLAUDE.md`, a map of the repository, what the agent is told to read and not read |
| Decomposition    | Several headless calls in `run.sh`: one for the repair, one for the report; fresh session or resumed   |
| Tools            | Tool allowlists, skills, hooks, subagents, scripts that parse the evidence                             |
| Verification     | Checks of your own that the agent runs before it stops                                                 |
| Caching          | Stable prompt prefixes, so later calls read from cache                                                 |
| Stopping         | Turn limits; stopping when your checks pass                                                            |

Try to run each change a few times to confirm its behavior - a flaky setup may score poorly on hidden tests.

Tune for the system, not the ticket. Your setup is graded on an incident you have not seen, so a prompt that walks the agent through this one is worth nothing. Do not put `tools/starter/` in front of your agent: it holds the answers to this ticket, and your practice results would stop meaning anything.

## Part III: Graded Run (70 pts)

Upload your `week6/` folder to the **Assignment 6 (Autograder)** assignment on Gradescope (not your whole repository). Zip it from inside the folder, for example `cd week6 && zip -r ../week6.zip .`, so hidden folders such as `.claude/` are included. The autograder runs your setup **once** on the held-out incident and scores the run:

| Component   | Points | Scoring                                                                                  |
| ----------- | ------ | ---------------------------------------------------------------------------------------- |
| Correctness | 35     | From the repair and impact scores                                                        |
| Cost        | 35     | Proportional: from 0 pts at/above the baseline to 35 pts at/below half the baseline cost |

The baseline is the provided `run.sh`, measured by staff on the held-out incident: **$1.50** per run.

You see the repair and impact scores, how many scenarios passed in each category, how long it took, the cost, the tokens by model, and any requests the gateway refused. You do not see the held-out incident, its scenarios, or your agent's output.

You can submit to the autograder **only three times**; Gradescope enforces the limit, and a fourth submission is not run. Each submission is graded with one run, so make sure your setup works in `tools/practice.py` before you spend one. Your active submission on Gradescope is the one that counts. A run that fails because the grader itself broke does not use up a submission.

**Leaderboard.** Gradescope shows whether your run passed and what it cost, under a name you choose. The lowest cost among passing submissions earns bonus points!

## Write-up (30 pts)

Fill in `writeup.md`, push it to your repository, and submit the repository link to the **Assignment 6 (Writeup)** assignment on Gradescope. You can resubmit the write-up as often as you like, so submit it after your last graded run and include that run's results. It covers Parts I to III and is graded on the quality of your experiments and reasoning, not on your final number: what you measured, what you changed and why, what happened, and what you would expect to carry over to an unseen incident.

## Rules

- A graded run is unattended. No step may wait for a person.
- Every model call goes through the gateway. Calling a model any other way, or shipping your own API key, is an honor code violation. So is sending the held-out task anywhere outside the container.
- Do not write the repair or the report yourself. `run.sh` has to get there by running an agent, and a passing run with no model calls is flagged and earns no cost credit. Submissions that carry copies of files from `app/` are reviewed.
- Staff read submissions and gateway logs.

## Deliverables

- **Assignment 6 (Autograder)** (up to 3 submissions): a zip of your `week6/` folder, with your `run.sh` and everything it needs.
- **Assignment 6 (Writeup)** (unlimited submissions): a link to your repository, with a completed `week6/writeup.md` pushed.

## Evaluation Rubric (100 pts total)

| Part                  | Points | What earns full credit                                                                   |
| --------------------- | ------ | ---------------------------------------------------------------------------------------- |
| Correctness           | 35     | The graded run correctly resolves the held-out incident                                  |
| Cost                  | 35     | Proportional: from 0 pts at/above the baseline to 35 pts at/below half the baseline cost |
| Write-up: baseline    | 10     | Baseline run recorded, with an account of where the tokens went                          |
| Write-up: experiments | 20     | Each change motivated & reported (including failures); reasoning about what transfers    |

## SUBMISSION INSTRUCTIONS

There are two Gradescope assignments for this week.

1. **Assignment 6 (Autograder): up to 3 submissions.** Zip and upload your `week6/` folder (see Part III). Each submission is graded live with one run on the held-out incident, and you cannot get more than three.
2. **Assignment 6 (Writeup): unlimited submissions.**
   1. Make sure you have all changes pushed to your remote repository for grading, including `week6/writeup.md` with your autograder results.
   2. **Make sure you've added `mihail911`, `isaackann`, and `vdaita` as collaborators on your assignment repository.**
   3. Submit the link to your repository on Gradescope.
