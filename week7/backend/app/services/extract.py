import re


def extract_action_items(text: str) -> list[str]:
    results: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\[\s*\]\s*", "", line)
        line = re.sub(r"^\[x\]\s*", "", line, flags=re.IGNORECASE)

        match = re.match(r"^(todo|task|fixme|action):\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            results.append(match.group(2).strip())
            continue

        if re.search(r"[.!?]$", line) or re.search(r"\b(todo|task|fixme|action)\b", line, flags=re.IGNORECASE):
            results.append(line)

    return results


