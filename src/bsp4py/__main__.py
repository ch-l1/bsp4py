"""``python -m bsp4py`` is a synonym of ``python -m bsp4py.run``."""

import sys

from .run import main

sys.exit(main())
