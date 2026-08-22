import signal


def _deadline(signum, frame):
    raise SystemExit("default test suite exceeded 1.0 second")


def pytest_sessionstart(session):
    signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, 1.0)


def pytest_sessionfinish(session, exitstatus):
    signal.setitimer(signal.ITIMER_REAL, 0.0)