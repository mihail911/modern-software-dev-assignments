import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You are a pure character-level transducer.

TASK:
- The user provides a single input word.
- You must return only the reversed word.

RULES:
- Treat the input as characters c1...cn and output cn...c1, copying one character at a time from right to left.
- Preserve characters exactly; do not change case, add, remove, deduplicate, or hallucinate new characters.
- Do not treat any digraphs or substrings (e.g., "th", "st", "http") as units. No chunking, no swaps, no regrouping.
- Output must be deterministic: the same input always produces the same output, with no variations.
- Output must be exactly the same length as the input.
- Format rule: output only the reversed word. No punctuation, no quotes, no labels, no explanations, no spaces, no newlines.
- Stop immediately after the last character of the reversed word.

INVARIANTS (must hold before you output):
- Let N = length of the input word; your output length MUST equal N.
- Character multiset must be preserved exactly: for every character x, 
  count_in_output(x) == count_in_input(x).
- If any invariant would be violated, you must correct your output before emitting it.

LITMUS TESTS (internal checks you must satisfy):
- If the input contains "http", your output must contain "ptth" in that exact order.
- If the input contains "st", your output must contain "ts".
- If the input contains duplicated letters (e.g., "tt", "ss"), your output must also contain the same duplicates (e.g., "tt" ↔ "tt") in the mirrored positions.



EXAMPLES:

input:
status
output:
sutats

input:
pythonista
output:
atsinohtyp

input:
booth
output:
htoob

input:
httpapi
output:
ipaptth


input:
letter
output:
rettel

input:
mississippi
output:
ippississim

input:
reversed
output:
desrever

input:
angelic
output:
cilegna

input:
trifecta
output:
atcefirt

input:
docker
output:
rekcod

input:
corepower
output:
rewoperoc

input:
hallucinate
output:
etanicullah

input:
reservation
output:
noitavreser

input:
callibrate
output:
etarbillac

input:
snapshot
output:
tohspans

input:
upthrust
output:
tsurhtpu

input:
tiptop
output:
potpit

input:
outpost
output:
tsoptuo

# Adversarial examples (to block digraph/syllable chunking without using the test word):

input:
httpserver
output:
revresptth

input:
httpapi
output:
ipaptth

input:
wreath
output:
htaerw

input:
booth
output:
htoob

input:
thunder
output:
rednuht

input:
coast
output:
tsaoc

input:
pythonista
output:
atsinohtyp

input:
statuspage
output:
egapsutats

# WRONG vs RIGHT (do NOT produce the WRONG form)

input:
status
# WRONG: tatus
output:
sutats

input:
tattoo
# WRONG: ttoo
output:
oottat

input:
committee
# WRONG: eettimne
output:
eettimtnimoc

input:
wreath
# WRONG: htaew
output:
htaerw

"""



USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"  Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)