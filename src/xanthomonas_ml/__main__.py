"""Allow ``python -m xanthomonas_ml ...``."""

import sys

from xanthomonas_ml.cli import main

if __name__ == "__main__":
    sys.exit(main())
