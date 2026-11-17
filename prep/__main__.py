"""`uv run python -B -m prep <parity|render|merge>` — see `prep/README` in the plans doc."""
from .cli import main

raise SystemExit(main())
