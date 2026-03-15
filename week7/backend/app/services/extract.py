import re


def extract_action_items(text: str) -> list[str]:
    lines = text.splitlines()
    results: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        normalized = stripped.lower()

        if re.match(r"^-\s*\[\s*\]\s+.+", stripped):
            task_text = re.sub(r"^-\s*\[\s*\]\s+", "", stripped)
            results.append(task_text)
        elif "todo:" in normalized or "action:" in normalized:
            clean_line = stripped.lstrip("- ")
            results.append(clean_line)
        elif stripped.endswith("!") and len(stripped) > 1:
            clean_line = stripped.lstrip("- ")
            results.append(clean_line)

    return results


def extract_hashtags(text: str) -> list[str]:
    pattern = r"#(\w+)"
    matches = re.findall(pattern, text)
    return list(set(matches))


def extract_all(text: str) -> dict[str, list[str]]:
    return {"action_items": extract_action_items(text), "hashtags": extract_hashtags(text)}
