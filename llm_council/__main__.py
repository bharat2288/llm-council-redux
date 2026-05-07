"""Allow `python -m llm_council ...` to dispatch to cli.main()."""
import sys

from llm_council.cli import main

if __name__ == "__main__":
    sys.exit(main())
