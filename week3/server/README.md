# Weather MCP Server (STDIO)

Local [Model Context Protocol](https://modelcontextprotocol.io) server named **`weather`** that exposes two tools for assistants (Claude Desktop, Cursor, etc.):

| Tool | Purpose |
|------|--------|
| `get_current_weather` | Current conditions for a city (plain English) |
| `get_forecast` | 1–7 day forecast (one line per day, plain English) |

Data comes from [Open-Meteo](https://open-meteo.com/) (free, no API key). Tool outputs are **human-readable strings**, not raw JSON.

## Requirements

- Python 3.10+
- Network access to `geocoding-api.open-meteo.com` and `api.open-meteo.com`

## Setup

From the **repository root** (`cs146s-assignments/`):

```bash
cd week3/server
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` only if you implement extra-credit API key support.

## Run (STDIO)

The server speaks JSON-RPC on **stdout**. Do not run it directly in a normal terminal for chatting; attach it from an MCP client.

**Manual smoke test (prints tool results; logs on stderr):**

```bash
uv run python -c "
import asyncio
from pathlib import Path
import importlib.util
p = Path('week3/server/main.py').resolve()
spec = importlib.util.spec_from_file_location('w', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
async def main():
    print(await m.get_current_weather('Berlin'))
asyncio.run(main())
"
```

## MCP Inspector (development)

From **repository root**, with `fastmcp` installed (included via `requirements.txt`):

```bash
uv run fastmcp dev inspector week3/server/main.py
```

This opens the MCP Inspector so you can connect, list tools, and invoke `get_current_weather` / `get_forecast`.

> Some docs use `mcp dev …`; that CLI comes from the `mcp` package with CLI extras. The equivalent here is **`fastmcp dev inspector`** on the server file.

## Claude Desktop / Cursor

Configure the host to launch the server as a subprocess, for example:

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": ["run", "python", "/absolute/path/to/cs146s-assignments/week3/server/main.py"],
      "cwd": "/absolute/path/to/cs146s-assignments"
    }
  }
}
```

Adjust `command`/`args` if you use a venv or `python3` directly.

## Tools (behavior summary)

- **`get_current_weather(city)`** — Geocodes the city, fetches `current_weather`, returns e.g.  
  `Current weather in London: 10.8°C, Mainly clear.`
- **`get_forecast(city, days=3)`** — `days` must be 1–7. Returns one line per day:  
  `2026-03-20: High 18°C / Low 10°C, Partly cloudy.`

Errors are **returned as strings** (empty city, unknown city, bad `days`, network failure, parse issues). The process does not crash on bad input.

## Logging

Application logs use `logging` to **stderr only** (`INFO` for calls/results, `ERROR` for failures). **Never use `print()`** in this server — stdout is reserved for MCP JSON-RPC.

## Example Conversation

```
User:    What's the weather in Tokyo?
Claude:  [calls get_current_weather(city="Tokyo")]
Server:  "Current weather in Tokyo: 12°C, Partly cloudy."
Claude:  It's currently 12°C and partly cloudy in Tokyo.

User:    Give me a 5-day forecast for Berlin.
Claude:  [calls get_forecast(city="Berlin", days=5)]
Server:  "2026-03-20: High 14°C / Low 6°C, Mainly clear.
          2026-03-21: High 16°C / Low 7°C, Partly cloudy.
          2026-03-22: High 11°C / Low 5°C, Slight rain.
          2026-03-23: High 9°C / Low 4°C, Overcast.
          2026-03-24: High 12°C / Low 6°C, Clear sky."
Claude:  Here's Berlin's 5-day forecast: Monday mostly clear at 14°C ...
```

## Error Handling

| Condition | Tool | Response |
|-----------|------|----------|
| Empty city string | both | `"City name cannot be empty. Provide a valid city name."` |
| City not found | both | `"City not found: {city}. Check spelling or try a nearby major city."` |
| `days` < 1 or > 7 | `get_forecast` | `"days must be between 1 and 7. You requested {days}."` |
| API timeout / unreachable | both | `"Weather service unavailable. Please try again shortly."` |
| Unexpected API response | both | `"Could not parse weather data for {city}."` |

The server never raises unhandled exceptions — every error path returns a plain English string.

## Deployment

The server runs as a subprocess spawned by the MCP client (Claude Desktop, Cursor).
No separate daemon or process manager is needed — the client starts it on demand.

To run persistently for local testing outside of a client:

```bash
# From cs146s-assignments/
.venv/bin/python week3/server/main.py
```

The process will block waiting for JSON-RPC input on stdin. Attach an MCP client
to interact with it.

## Tests (manual)

See the assignment "TEST CASES" section: connect in Inspector, verify both tools, London/Tokyo/Paris scenarios, empty city, invalid `days`, fake city, then a valid city again to confirm the server stays up.

## Dependencies

See `requirements.txt`:

- `fastmcp>=3.1.0`
- `httpx>=0.27.0`
- `python-dotenv>=1.0.0` (reserved for optional `.env` / extra credit)
