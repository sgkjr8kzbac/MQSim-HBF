"""Small shared helpers (CSV/JSON output and overlap parsing)."""
import csv
import json


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def write_table(path, rows):
    """CSV of plain Python values (numpy scalars converted; None written empty)."""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _plain(value) for key, value in row.items()} for row in rows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_overlap(text):
    """'serial', 'ideal' or 'prefetch:<d>' (reads run up to d layers ahead) -> (mode, d or None)."""
    if text in ("serial", "ideal"):
        return text, None
    mode, _, depth = str(text).partition(":")
    if mode == "prefetch" and depth.isdigit():
        return mode, int(depth)
    raise ValueError(f"Invalid overlap {text!r}: use serial, ideal or prefetch:<d>")
