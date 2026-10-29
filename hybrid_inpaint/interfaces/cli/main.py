"""Console entry point for ``hybrid-inpaint``."""

from __future__ import annotations

from hybrid_inpaint.interfaces.cli.commands import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
