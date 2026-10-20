
import numpy as np

def michel_criterion(Re_theta,Re_x): 

    mc = 1.174 * (1+22400/Re_x) * Re_x**(0.46)
    idx = np.where(Re_theta >= mc)[0]
    if len(idx) > 0:
        transition_index = idx[0]
        transition_flag = True
    else:
        transition_index = None
        transition_flag = False
        
    return  transition_index , transition_flag