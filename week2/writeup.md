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
```
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
```

### Exercise 2: Add Unit Tests
Prompt: 
```
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
```

### Exercise 3: Refactor Existing Code for Clarity
Prompt: 
```
Perform a refactor of the code in the backend, focusing in particular on
well-defined API contracts/schemas, database layer cleanup, app
lifecycle/configuration, error handling
```

Generated/Modified Code Snippets:
```
New files
- week2/app/config.py (whole file): centralised frozen-dataclass Settings built
  from environment variables (APP_DATA_DIR/APP_DB_PATH, OLLAMA_MODEL, etc.) with
  a cached get_settings() accessor; replaces scattered module-level constants.
- week2/app/schemas.py (whole file): Pydantic v2 request/response models
  (NoteCreate, Note, ActionItem, ExtractRequest/Response, ActionItemDoneUpdate,
  ExtractedItems, ErrorResponse) so routes have explicit contracts.
- week2/app/errors.py (whole file): domain exceptions (AppError, NotFoundError,
  BadRequestError, UpstreamServiceError) and register_exception_handlers(), which
  normalises app errors, validation errors and unexpected errors into one JSON
  envelope.

Rewritten files
- week2/app/main.py (whole file): added an async lifespan() that runs init_db()
  on startup (no more import-time side effects) and a create_app() factory that
  wires routers, exception handlers, the index route and static files.
- week2/app/db.py (whole file): replaced the raw-connection helper with a
  @contextmanager get_connection() that commits/rolls back/closes, enables
  PRAGMA foreign_keys, and added typed NoteRecord/ActionItemRecord dataclasses so
  callers no longer touch sqlite3.Row. Schema now declares FK ON DELETE CASCADE
  and an index on action_items(note_id); mark_action_item_done() returns whether
  a row was updated.
- week2/app/routers/action_items.py (whole file): typed request/response models,
  response_model declarations, 404 on missing action item, removed ad-hoc
  Dict[str, Any] parsing.
- week2/app/routers/notes.py (whole file): typed models, response_model, 201 on
  create, 404 via NotFoundError.
- week2/app/services/extract.py: removed a stray debug print(), derived the Ollama
  structured-output schema from the ExtractedItems pydantic model, read model and
  temperature from settings, and raised UpstreamServiceError instead of leaking
  ValueError/ollama errors. Public function names kept stable for tests.
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