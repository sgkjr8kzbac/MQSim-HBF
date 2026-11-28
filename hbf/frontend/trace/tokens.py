"""Token-level logical access schema and CSV I/O (access.csv)."""
import csv
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Access:
    step: int
    layer: int
    kv_head: int
    kind: str
    op: str
    token_start: int
    token_end: int
    bits: int = 16

    def validate(self, layout):
        """``kind`` names a tensor of the layout and ``bits`` must match its stored format."""
        shape = layout.shape
        numbers = (self.step, self.layer, self.kv_head, self.token_start, self.token_end, self.bits)
        spec = layout.specs.get(self.kind)
        if any(type(n) is not int for n in numbers) or spec is None or not (
            self.step >= -1 and 0 <= self.layer < shape.layers and
            0 <= self.kv_head < shape.kv_heads and
            self.op in {"R", "W", "H"} and layout.windows[self.kind][0] <= self.token_start < self.token_end
            <= layout.windows[self.kind][1] and self.bits == spec.bits
        ) or (hasattr(layout, "reads") and not layout.reads(self.layer, self.kind, self.kv_head)):
            raise ValueError(f"Invalid logical access: {self}")


def write_access(path, accesses):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f.name for f in fields(Access)])
        writer.writerows(vars(access).values() for access in accesses)


def read_access(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = [field.name for field in fields(Access)]
        if reader.fieldnames != names:
            raise ValueError("Invalid logical access CSV header")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("Invalid logical access CSV row")
            yield Access(**{key: val if key in {"op", "kind"} else int(val) for key, val in row.items()})
