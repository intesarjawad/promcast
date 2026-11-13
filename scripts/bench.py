"""Run the same benchmark command as the installed CLI."""

import sys

from promcast.shell.cli import main

if __name__ == "__main__":
    sys.argv = ["promcast", "bench", *sys.argv[1:]]
    raise SystemExit(main())
