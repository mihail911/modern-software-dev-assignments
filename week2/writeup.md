# Week 2 Write-up
Tip: To preview this markdown file
- On Mac, press `Command (⌘) + Shift + V`
- On Windows/Linux, press `Ctrl + Shift + V`

## INSTRUCTIONS

Fill out all of the `TODO`s in this file.

## SUBMISSION DETAILS

Name: **TODO** \
SUNet ID: **TODO** \
Citations: **TODO**

This assignment took me about **TODO** hours to do. 


## YOUR RESPONSES
For each exercise, please include what prompts you used to generate the answer, in addition to the location of the generated response. Make sure to clearly add comments in your code documenting which parts are generated.

### Exercise 1: Scaffold a New Feature
Prompt: 
i want you to implement extract_action_items_llm(text: str). the function should use Ollama's Python library (or HTTP API) to extract action items from the text. I want you to use ollama3.1:8b in that function. Use structured output to return a JSON list of strings. Reference the existing extract_action_items for context. Use this documentation https://ollama.com/blog/structured-outputs to make the function.

TODO
``` 

Generated Code Snippets:
File: week2/app/services/extract.py

Line 14-16:
Add class ActionItemList(BaseModel) for output JSON array of strings 

Line 9:
Add relevant import like import BaseModel

Line 82-111:
Add function extract_action_items_llm(text: str). 
This function will send a prompt to ollama3.1:8 model using 'format' parameter with the Pydantic schema of ActionItemList, and also validates the result using model_validate_json. 
TODO: List all modified code files with the relevant line numbers.
```

### Exercise 2: Add Unit Tests
Prompt: 
i want you to make unit testing for function extract_action_items_llm. Test cases includes normal list of todos (bullet lists), keyword-prefixed lines, and empty input. Use unittest.mock to mock the ollama.chat response so I don't need the actual model running during tests.

TODO
``` 

Generated Code Snippets:
Add new import like import patch, import MagicMock, and import json
Add function test_extract_action_items_llm_with_todos that test input with bullet list.
Add function test_extract_action_items_llm_without_tasks that test input with keyword-prefixed lines (todo:, action:, next:)
Add function test_extract_action_items_llm_empty_string that test empty input 

TODO: List all modified code files with the relevant line numbers.
```

### Exercise 3: Refactor Existing Code for Clarity
Prompt: 
```
TODO
``` 

Generated/Modified Code Snippets:
```
TODO: List all modified code files with the relevant line numbers. (We anticipate there may be multiple scattered changes here – just produce as comprehensive of a list as you can.)
```


### Exercise 4: Use Agentic Mode to Automate a Small Task
Prompt: 
```
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
```


### Exercise 5: Generate a README from the Codebase
Prompt: 
```
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
```


## SUBMISSION INSTRUCTIONS
1. Hit a `Command (⌘) + F` (or `Ctrl + F`) to find any remaining `TODO`s in this file. If no results are found, congratulations – you've completed all required fields. 
2. Make sure you have all changes pushed to your remote repository for grading.
3. Submit via Gradescope. 