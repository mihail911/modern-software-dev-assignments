# Week 1 — Prompting Techniques

You will practice multiple prompting techniques by crafting prompts to complete specific tasks. Each task’s instructions are at the top of its corresponding source file.

## Installation
Make sure you have first done the installation described in the top-level `README.md`. 

## Ollama installation
We will be using a tool to run different state-of-the-art LLMs locally on your machine called [Ollama](https://ollama.com/). Use one of the following methods:

- macOS (Homebrew):
  ```bash
  brew install --cask ollama 
  ollama serve
  ```

- Linux (recommended):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- Windows:
  Download and run the installer from [ollama.com/download](https://ollama.com/download).

Verify installation:
```bash
ollama -v
```

Before running the test scripts, make sure you have the following models pulled. You only need to do this once (unless you remove the models later):
```bash
ollama run mistral-nemo:12b
ollama run llama3.1:8b
```

## Techniques and source files
- [x] K-shot prompting — `week1/k_shot_prompting.py`
- [x] Chain-of-thought — `week1/chain_of_thought.py`
- [x] Tool calling — `week1/tool_calling.py`
- [ ] Self-consistency prompting — `week1/self_consistency_prompting.py`
- [ ] RAG (Retrieval-Augmented Generation) — `week1/rag.py`
- [ ] Reflexion — `week1/reflexion.py`

## Deliverables
- Read the task description in each file.
- Design and run prompts (look for all the places labeled `TODO` in the code). That should be the only thing you have to change (i.e. don't tinker with the model). 
- Iterate to improve results until the test script passes.
- Save your final prompt(s) and output for each technique.
- Make sure to include in your submission the completed code for each prompting technique file. ***Double check that all `TODO`s have been resolved.***

## Evaluation rubric (60 pts total)
- 10 for each completed prompt across the 6 different prompting techniques