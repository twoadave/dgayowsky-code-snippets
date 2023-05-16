"""
Created on Mon May 15 2023

@author: David Gayowsky

Simple PyMeep Monte-Carlo "learning" simulation. Replicated from Viktor's work.
"""

#######################################################################

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video 
import os
import scipy as sp
from scipy.constants import c
import copy

#######################################################################