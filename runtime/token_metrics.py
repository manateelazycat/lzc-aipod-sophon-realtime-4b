"""Cumulative observations of real inference; no tokenization or GPU operations."""
import math
import threading


class TokenMetrics:
    NAMES = {"input": "prompt_tokens_total", "output": "generation_tokens_total",
             "requests": "requests_total", "work": "work_total"}

    def __init__(self, *known):
        self.lock = threading.Lock()
        self.values = {name: 0 for name in known}
        self.unknown = set()

    def add(self, **values):
        # A broken observer cannot fail a successful inference or invent zero.
        with self.lock:
            for name, value in values.items():
                if name not in self.NAMES or name in self.unknown:
                    continue
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 0
                        or (name != "work" and int(value) != value)):
                    self.values.pop(name, None)
                    self.unknown.add(name)
                    continue
                self.values[name] = self.values.get(name, 0) + value

    def render(self):
        with self.lock:
            values = dict(self.values)
        return "".join(f"# TYPE aipod:{self.NAMES[name]} counter\n"
                       f"aipod:{self.NAMES[name]} {value}\n" for name, value in values.items())
