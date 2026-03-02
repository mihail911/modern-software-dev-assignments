Collecting workspace information# Action Item Extractor

An AI-powered FastAPI application that intelligently extracts and manages action items from free-form notes using Ollama's large language models. This project combines heuristic-based extraction with LLM-powered alternatives for flexible, accurate task identification.

## Project Overview

The Action Item Extractor is a modern web application designed to help users convert unstructured notes into organized, actionable tasks. It features:

- **Dual extraction modes**: Heuristic-based extraction (fast, no dependencies) and LLM-powered extraction using Ollama
- **REST API**: FastAPI backend with complete CRUD operations for notes and action items
- **Data persistence**: SQLite database for storing notes and extracted action items
- **Web frontend**: Minimal, responsive HTML interface for intuitive interaction
- **Comprehensive testing**: Unit tests with mocked LLM responses for reliable CI/CD

## Table of Contents

- Prerequisites
- Setup & Installation
- Configuration
- Running the Application
- API Endpoints
- Testing
- Project Structure

## Prerequisites

- **Python 3.8+**
- **Poetry** (for dependency management)
- **Ollama** with `llama3.1:8b` model installed and running
- **Git** (for version control)

### Installing Ollama

1. Download Ollama from [ollama.com](https://ollama.com)
2. Install and start the Ollama service
3. Pull the required model:
   ```bash
   ollama pull llama3.1:8b
   ```
4. Verify the model is available:
   ```bash
   ollama list
   ```

The Ollama service runs on `http://localhost:11434` by default.

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd modern-software-dev-assignments/week2
```

### 2. Create and Activate Virtual Environment

**Using conda:**
```bash
conda create -n cs146s python=3.11
conda activate cs146s
```

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
poetry install
```

If Poetry is not installed, install it first:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 4. Environment Configuration

Create a `.env` file in the project root (optional for default settings):

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=INFO
```

## Running the Application

### Start the FastAPI Server

From the project root, run:

```bash
poetry run uvicorn week2.app.main:app --reload
```

**Options:**
- `--reload`: Enable auto-reload on code changes (development only)
- `--host 0.0.0.0`: Bind to all network interfaces
- `--port 8000`: Specify custom port (default: 8000)

The application will be available at: **http://127.0.0.1:8000/**

### Access the Web Interface

Open your browser and navigate to `http://127.0.0.1:8000/` to access the interactive frontend.

## API Endpoints

All endpoints are JSON-based REST APIs. The application exposes two main router groups:

### Notes Router (`/notes`)

#### Create Note
```http
POST /notes
Content-Type: application/json

{
  "content": "Meeting notes about the project..."
}
```

**Response (201):**
```json
{
  "id": 1,
  "content": "Meeting notes about the project...",
  "created_at": "2024-01-15T10:30:00"
}
```

**Error (400):** Content is required

---

#### List All Notes
```http
GET /notes
```

**Response (200):**
```json
[
  {
    "id": 1,
    "content": "Meeting notes...",
    "created_at": "2024-01-15T10:30:00"
  }
]
```

---

#### Get Single Note
```http
GET /notes/{note_id}
```

**Response (200):**
```json
{
  "id": 1,
  "content": "Meeting notes...",
  "created_at": "2024-01-15T10:30:00"
}
```

**Error (404):** Note not found

---

### Action Items Router (`/action-items`)

#### Extract Action Items
```http
POST /action-items/extract
Content-Type: application/json

{
  "text": "- [ ] Set up database\n* Implement API endpoint\n1. Write tests",
  "save_note": true
}
```

**Response (200):**
```json
{
  "note_id": 1,
  "items": [
    {
      "id": 1,
      "text": "Set up database"
    },
    {
      "id": 2,
      "text": "Implement API endpoint"
    },
    {
      "id": 3,
      "text": "Write tests"
    }
  ]
}
```

**Behavior:**
- Uses `extract_action_items_llm()` by default (falls back to heuristic extraction if LLM unavailable)
- If `save_note` is `true`, the original text is saved as a note
- Supports bullet points (`-`, `*`, `•`), numbered lists (`1.`), checkboxes (`[ ]`), and keyword prefixes (`todo:`, `action:`, `next:`)

---

#### List Action Items
```http
GET /action-items
GET /action-items?note_id=1
```

**Response (200):**
```json
[
  {
    "id": 1,
    "note_id": 1,
    "text": "Set up database",
    "done": false,
    "created_at": "2024-01-15T10:30:00"
  }
]
```

---

#### Mark Action Item as Done
```http
POST /action-items/{action_item_id}/done
Content-Type: application/json

{
  "done": true
}
```

**Response (200):**
```json
{
  "id": 1,
  "text": "Set up database"
}
```

**Error (404):** Action item not found

---

## Testing

### Run the Full Test Suite

```bash
poetry run pytest tests/
```

### Run Tests with Verbose Output

```bash
poetry run pytest tests/ -v
```

### Run Specific Test File

```bash
poetry run pytest tests/test_extract.py -v
```

### Run Tests with Coverage Report

```bash
poetry run pytest tests/ --cov=app --cov-report=html
```

### Test Files

- **test_extract.py** - Tests for action item extraction functions:
  - `test_extract_bullets_and_checkboxes()` - Heuristic extraction with various bullet formats
  - `test_extract_action_items_llm_with_bullet_list()` - LLM extraction with bullet lists (mocked)
  - `test_extract_action_items_llm_with_keyword_prefixes()` - LLM extraction with keyword-prefixed lines
  - `test_extract_action_items_llm_empty_input()` - LLM extraction with empty input

All LLM tests use `unittest.mock.patch` to mock the Ollama API response, ensuring tests run without requiring the actual model.

## Project Structure

```
week2/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization & routing
│   ├── db.py                   # SQLite database layer
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── notes.py           # Notes CRUD endpoints
│   │   └── action_items.py    # Action items extraction & management
│   └── services/
│       ├── __init__.py
│       └── extract.py         # Business logic for action item extraction
├── frontend/
│   └── index.html             # Web UI
├── tests/
│   ├── __init__.py
│   └── test_extract.py        # Unit tests
├── data/                      # SQLite database storage (auto-created)
├── .env                       # Environment variables (optional)
├── pyproject.toml             # Poetry dependencies & configuration
└── README.md                  # This file
```

## Key Features

### Extraction Modes

#### 1. Heuristic Extraction (Fast, No Dependencies)
- Recognizes bullet points (`-`, `*`, `•`)
- Detects numbered lists (`1.`, `2.`, etc.)
- Identifies checkboxes (`[ ]`, `[todo]`)
- Recognizes keyword prefixes (`todo:`, `action:`, `next:`)
- Fallback for imperative sentences (starts with verbs like "Fix", "Implement", "Review")

#### 2. LLM-Powered Extraction (Accurate, Ollama-Dependent)
- Uses `llama3.1:8b` model via Ollama
- Structured JSON output via Pydantic schema
- Intelligently reformats action items (removes bullets, ensures verb-first formatting)
- Automatic fallback to heuristic extraction if LLM is unavailable
- Comprehensive error handling and logging

### Data Persistence

- **SQLite database** automatically created in app.db
- **Notes table**: Stores original text with timestamps
- **Action items table**: Stores extracted tasks with completion status and note association
- **Foreign key relationships** maintain data integrity

### Error Handling

- Graceful fallbacks when Ollama is unavailable
- Comprehensive validation of API inputs
- Detailed error messages for debugging
- Structured exception handling throughout the codebase

## Troubleshooting

### Ollama Connection Error
```
Error: Failed to connect to ollama3.1:8b
```
**Solution:** Ensure Ollama is running and the model is pulled:
```bash
ollama run llama3.1:8b
```

### Port Already in Use
```
Address already in use
```
**Solution:** Use a different port:
```bash
poetry run uvicorn week2.app.main:app --port 8001 --reload
```

### No Tests Found
```
ERROR collecting test session
```
**Solution:** Ensure you're running pytest from the project root and pytest is installed:
```bash
poetry install
poetry run pytest tests/ -v
```

## Development Notes

- The frontend uses vanilla JavaScript with no external dependencies
- All API responses include proper HTTP status codes and error messages
- The application initializes the SQLite database automatically on startup
- Logging is configured for production debugging (check console output)

## License

This project is part of Stanford CS146S coursework.