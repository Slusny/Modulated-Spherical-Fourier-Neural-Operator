'''
This module is used to log the progress of the model inference. 
Used in run methods of sfno and fourcastnet models.
'''

import logging
import time

from climetlab.utils.humanize import seconds

LOG = logging.getLogger(__name__)


class Stepper:
    def __init__(self, step, lead_time):
        self.step = step
        self.lead_time = lead_time
        self.start = time.time()
        self.last = self.start
        self.num_steps = lead_time // step
        LOG.info("Starting inference for %s steps (%sh).", self.num_steps, lead_time)

    def __enter__(self):
        return self

    def __call__(self, i, step):
        now = time.time()
        elapsed = now - self.start
        speed = (i + 1) / elapsed
        eta = (self.num_steps - i) / speed
        LOG.info(
            "Done %s out of %s in %s (%sh), ETA: %s.",
            i + 1,
            self.num_steps,
            seconds(now - self.last),
            step,
            seconds(eta),
        )
        self.last = now

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("Elapsed: %s.", seconds(elapsed))
        LOG.info("Average: %s per step.", seconds(elapsed / self.num_steps))
