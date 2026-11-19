"""How the supported harnesses are pointed at the gateway.

The grader and ``practice.py`` both build a run's environment here, so a
``run.sh`` that works in practice sees the same variables when it is graded.
"""

import json
import shutil
from pathlib import Path

PLACEHOLDER_KEY = "cs146s-gateway"  # the gateway swaps in the real credential


def gateway_env(gateway_url: str, home: Path, give_keys: dict) -> dict:
    """Environment variables for one run.

    ``give_keys`` says, per provider, whether to hand the harness a placeholder
    key. That is right whenever the gateway holds the real one. It is wrong
    for a subscription login, where the harness must send its own credentials.
    """
    env = {
        "CS146S_GATEWAY_URL": gateway_url,
        # Claude Code
        "ANTHROPIC_BASE_URL": f"{gateway_url}/anthropic",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        # Codex reads its provider from $CODEX_HOME/config.toml (written below)
        "CODEX_HOME": str(home / ".codex"),
        # Anything else that speaks the OpenAI API
        "OPENAI_BASE_URL": f"{gateway_url}/openai/v1",
    }
    if give_keys.get("anthropic"):
        env["ANTHROPIC_API_KEY"] = PLACEHOLDER_KEY
    if give_keys.get("openai"):
        env["OPENAI_API_KEY"] = PLACEHOLDER_KEY
        env["CS146S_GATEWAY_KEY"] = PLACEHOLDER_KEY
    return env


def write_codex_config(home: Path, gateway_url: str, login: Path = None) -> Path:
    """A Codex provider that sends Responses API calls through the gateway.

    With ``login`` (a ChatGPT login's ``auth.json``, practice only) Codex uses that
    login and the gateway forwards it to the ChatGPT backend, metering every response.
    """
    path = home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if login is not None:
        shutil.copy(login, home / ".codex" / "auth.json")
        path.write_text(
            'model_provider = "cs146s"\n'
            "\n"
            "[model_providers.cs146s]\n"
            'name = "CS146S gateway (ChatGPT login)"\n'
            f'base_url = "{gateway_url}/chatgpt"\n'
            'wire_api = "responses"\n'
            "requires_openai_auth = true\n"
        )
        return path
    path.write_text(
        'model_provider = "cs146s"\n'
        "\n"
        "[model_providers.cs146s]\n"
        'name = "CS146S gateway"\n'
        f'base_url = "{gateway_url}/openai/v1"\n'
        'env_key = "CS146S_GATEWAY_KEY"\n'
        'wire_api = "responses"\n'
    )
    return path


def write_droid_config(home: Path, gateway_url: str, card: dict) -> Path:
    """Droid custom models, one per rate-card model, in rate-card order.

    Droid names a custom model ``custom:<displayName>-<index>``, so the first
    model on the rate card is ``custom:claude-sonnet-5-5-0`` and so on.
    """
    base = {"anthropic": f"{gateway_url}/anthropic", "openai": f"{gateway_url}/openai/v1"}
    models = [
        {
            "model": name,
            "displayName": name,
            "provider": rates["provider"],
            "baseUrl": base[rates["provider"]],
            "apiKey": PLACEHOLDER_KEY,
        }
        for name, rates in card["models"].items()
    ]
    path = home / ".factory" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"customModels": models}, indent=2) + "\n")
    return path
