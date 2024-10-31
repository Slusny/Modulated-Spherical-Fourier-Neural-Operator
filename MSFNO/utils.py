
import time
import logging
import os
import numpy as np
from climetlab.utils.humanize import seconds
LOG = logging.getLogger(__name__)


class Timer:
    '''
    A utility class to measure runtime.
    Envelop code in a with Timer('some title') statement to measure the time it takes to execute.
    '''
    def __init__(self, title,divisor=1.):
        self.title = title
        self.start = time.time()
        self.divisor = divisor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        LOG.info("%s: %s.", self.title, seconds(elapsed/self.divisor))
        print("%s: %s." % (self.title, seconds(elapsed/self.divisor)))


class FinTraining(Exception):
    "Raised if Training is finished"
    def __init__(self,message):
        super().__init__(message)
    
class Attributes():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  

class LocalLog():
    def __init__(self,do_local_logging,save_path=None):
        self.do_local_logging = do_local_logging
        self.save_path = save_path
        self.log_dict = {}
    
    def log(self,log):
        if self.do_local_logging:
            for k in log.keys():
                if k not in self.log_dict:
                    self.log_dict[k] = []
                self.log_dict[k].append(log[k])

    def save(self,file):
        if self.do_local_logging:
            save_dict = {}
            for k in self.log_dict.keys():
                save_dict[k] = np.array(self.log_dict[k])
            save_file = os.path.join(self.save_path,file)    
            np.save(save_file,save_dict)