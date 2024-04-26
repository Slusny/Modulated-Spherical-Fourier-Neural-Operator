import os
import numpy as np
import glob

cp_list = list(sorted(glob.glob(os.path.join(args.eval_checkpoint_path,"checkpoint_*")),key=len))