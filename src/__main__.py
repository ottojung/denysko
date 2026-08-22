import sys

from .denysko import run


def entry() -> None:
    raise SystemExit(run(sys.argv[1:]))
