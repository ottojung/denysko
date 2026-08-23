import signal

import matplotlib.font_manager  # noqa: F401  (front-load heavy imports)
import numpy  # noqa: F401
import scipy.optimize  # noqa: F401
import src.denysko  # noqa: F401


def _deadline(signum, frame):
    raise SystemExit("default test suite exceeded 1.0 second")


def pytest_sessionstart(session):
    signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, 1.0)


def pytest_sessionfinish(session, exitstatus):
    signal.setitimer(signal.ITIMER_REAL, 0.0)