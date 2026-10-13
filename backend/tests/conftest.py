import logging

import pytest


@pytest.fixture(autouse=True)
def restore_the_askact_logger():
    """`api.configure_logging` (called by `build_app`) installs a handler on the `askact` logger. It is global state:
    left in place, a handler bound to one test's captured stderr would be written to by later tests, after pytest has
    closed that stream. Every test therefore starts and ends with the logger as it found it."""
    logger = logging.getLogger("askact")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
