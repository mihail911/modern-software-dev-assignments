# Assignment 6 (Writeup)

## Part I: Baseline

Run the provided `run.sh` on the starter incident (`python tools/practice.py`). Record the result below; add rows if you repeat it.

| Run | Repair | Impact | Cost | Model requests | Time |
| --- | ------ | ------ | ---- | -------------- | ---- |
| 1   | TODO   | TODO   | TODO | TODO           | TODO |

**Tokens by kind**, averaged if you ran the baseline more than once:

| Model | Input | Output | Cache read | Cache write | Cost |
| ----- | ----- | ------ | ---------- | ----------- | ---- |
| TODO  | TODO  | TODO   | TODO       | TODO        | TODO |

**Where did the money go?** Which kind of token dominated, and what in the run produced it:

> TODO

**What did the baseline get wrong?** Record scenarios passed per category and the impact report's precision and recall (averaged if you repeated the run). Then read the transcripts: what did the agent believe, and where did that belief come from?

| Category | Passed |
| -------- | ------ |
| TODO     | TODO   |

> TODO

## Part II: Experiments

One row per change you tried, including the ones you dropped. Try to run each change a few times to check its reliability; report how many runs you made.

| #   | Change | Why you expected it to help | Runs passed / attempted | Avg repair / impact | Avg cost | Kept? |
| --- | ------ | --------------------------- | ----------------------- | ------------------- | -------- | ----- |
| 1   | TODO   | TODO                        | TODO                    | TODO                | TODO     | TODO  |
| 2   | TODO   | TODO                        | TODO                    | TODO                | TODO     | TODO  |
| 3   | TODO   | TODO                        | TODO                    | TODO                | TODO     | TODO  |

**The change that fixed the most failing categories**, and why it worked:

> TODO

**The change that saved the most money**, and why it worked:

> TODO

**A change that did not work**, and what it showed you:

> TODO

**Your final setup.** What `run.sh` does, step by step, and which model handles each step:

> TODO

**Final setup on the starter incident.** Record your results; add rows for repeated runs. Briefly explain any variation you observed.

| Run | Repair | Impact | Cost |
| --- | ------ | ------ | ---- |
| 1   | TODO   | TODO   | TODO |

**What will transfer?** The held-out task is a different incident in Switchyard. Which parts of your setup are about how to work in this repository and its evidence, and which might only have helped with duplicate events:

> TODO

## Part III: Graded Submissions

Upload a zip of your `week6/` folder to the **Assignment 6 (Autograder)** assignment on Gradescope (up to 3 submissions, one run each) and report your autograder scores.

| Submission (up to 3, one run each) | Repair | Impact | Passed? | Cost | What you changed before it |
| ---------------------------------- | ------ | ------ | ------- | ---- | -------------------------- |
| 1                                  | TODO   | TODO   | TODO    | TODO | n/a                        |
| 2                                  | TODO   | TODO   | TODO    | TODO | TODO                       |
| 3                                  | TODO   | TODO   | TODO    | TODO | TODO                       |

**Starter incident versus held-out incident.** Did repair, impact and cost carry over? If not, give your best explanation, accounting for each submission having only one graded run:

> TODO

## Submission

1. `Command (⌘) + F` for `TODO`. No results means you're done. Delete rows you did not need.
2. Push all changes to your remote repository and submit the repository link to the **Assignment 6 (Writeup)** assignment on Gradescope. You can resubmit it as often as you like; your `week6/` folder goes to the separate **Assignment 6 (Autograder)** assignment (up to 3 submissions).
