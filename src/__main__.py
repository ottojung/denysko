import sys


def entry() -> None:
    args = sys.argv[1:]
    if len(args) != 1 or len(args[0]) != 1 or not "A" <= args[0] <= "Z":
        raise SystemExit(2)
    print(f"denysko: letter {args[0]} accepted (fitting not implemented yet)")
