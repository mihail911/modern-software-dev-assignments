import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
Take the user request and output a one word response.
The user is, effectively, providing a single input word at the end of their prompt. 
You should always treat that single word as an input, and return a one-word output.
You should always reverse the order of the letters in the input word.
Preserve the input characters exactly; do not change case, add, remove, deduplicate, or reorder beyond exact reversal.
Output must be a single word with no spaces, punctuation, or quotes.
Here are several example inputs and outputs:

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
"""

# input:


# output:



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