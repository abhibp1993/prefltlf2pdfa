class DSLError(ValueError):
    def __init__(self, message: str, line: int | None = None, suggestion: str | None = None):
        self.msg = message
        self.line = line
        self.suggestion = suggestion
        super().__init__(self._format())

    def _format(self) -> str:
        parts = []
        if self.line is not None:
            parts.append(f"Line {self.line}:")
        msg = self.msg
        if self.suggestion:
            if not msg.endswith("."):
                msg += "."
            parts.append(msg)
            parts.append(f"Did you mean '{self.suggestion}'?")
        else:
            parts.append(msg)
        return " ".join(parts)

    def __str__(self) -> str:
        return self._format()
