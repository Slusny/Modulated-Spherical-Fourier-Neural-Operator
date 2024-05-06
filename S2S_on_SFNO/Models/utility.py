
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

