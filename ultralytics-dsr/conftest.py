# Ensures pytest resolves `import ultralytics` to this project's vendored tree
# (pinned 8.3.247), not the environment's site-packages copy.

import os

# This conda env ships polars>=1.x whose CPU-flag parser raises a false-positive
# RuntimeError ("unknown feature flag: 'sse3'") on some CPUs, which aborts
# trainer.save_model() at the end of every run. Bypass it for the test suite.
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
