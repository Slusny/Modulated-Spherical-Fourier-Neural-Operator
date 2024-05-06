
import time
import logging
import os
from climetlab.utils.humanize import seconds
LOG = logging.getLogger(__name__)

def test_autoregressive_forecast(checkpoint_list,hyperparams):
    for checkpoint in checkpoint_list:
        print(f"Testing checkpoint {checkpoint}")
        model = S2SModel(hyperparams)
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        model = model.to(device)
        model.autoregressive_forecast()
        print("Test passed")


class Timer:
    '''
    A utility class to measure runtime.
    Envelop code in a with Timer('some title') statement to measure the time it takes to execute.
    '''
    def __init__(self, title):
        self.title = title
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("%s: %s.", self.title, seconds(elapsed))

