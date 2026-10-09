"""Hypothesis profiles for the property-based tests.

``dev`` keeps local runs quick. ``ci`` draws more examples, derandomizes so
the same inputs run on every push (a lucky seed cannot fail CI), and drops
the per-example deadline because scipy root-finding and AAA fits jitter on
shared runners. Pick one with the ``HYPOTHESIS_PROFILE`` environment variable.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=50)
settings.register_profile(
    "ci",
    max_examples=200,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
