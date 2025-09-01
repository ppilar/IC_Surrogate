# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path


from ice.utils_eval import evaluate_ddpm


if len(sys.argv) > 1:
    name_ddpm = sys.argv[1]
    eval_opt = sys.argv[2]
else:
    name_ddpm = 'ddpm_Jul02'
    E_opt = ''
    eval_opt = 'samples'
    print('automatically selecting models!')
    
print('ddpm:', name_ddpm, 'opt:', eval_opt)

npath = '../nets/'
evaluate_ddpm(npath, name_ddpm, E_opt = E_opt, eval_opt = eval_opt)

    
    
