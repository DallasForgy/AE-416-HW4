import numpy as np  
from scipy.interpolate import interp1d
from scipy.integrate import odeint
  
def solve_thwaites_BL(nu,l,Re_L,x_i,Ve_i,dVe_i,theta_0):  
    ''' Solved intergral boundary layer equations for laminar flow  
    '''         

    x   = x_i
    Ve  = getVe(x, x_i, Ve_i)
    dVe = getdVe(x, x_i, dVe_i)

    y0    = theta_0**2 * Ve[0]**6
    y     = odeint(odefcn, y0, x, args=(nu, x_i, Ve_i)).flatten()
    theta = np.sqrt(y / Ve**6)

    lambda_val = theta**2 * dVe / nu
    H          = getH(lambda_val)
    Re_theta   = Ve * theta / nu
    Re_x       = Ve * x / nu
    cf         = getcf(lambda_val, Re_theta)
    delta_star = H * theta
    delta      = 5.2 * x / np.sqrt(Re_x)

    return H,delta_star,delta,cf,theta,Re_x,Re_theta


def getH(lambda_val ): 
    """ Computes the shape factor, H

    Assumptions:
    None

    Source:
    None

    Inputs: 
    lamdda_val  - thwaites separation criteria [unitless]

    Outputs:  
    H           - shape factor [unitless]

    Properties Used:
    N/A
    """       
    
    H = np.where(lambda_val < 0, 
                 2.088 + 0.0731 / (0.14 + lambda_val),
                 2.61 - 3.75 * lambda_val + 5.24 * lambda_val**2)

    return H 
    
def odefcn(y,x, nu,x_i,Ve_i):
    """ Computes boundary layer functions using SciPy ODE solver 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    y           - initial conditions of functions    [unitless]
    x           - new x values at which to solve ODE [unitless]
    nu          - kinematic viscosity                [m^2/s]
    x_i         - intial array of x values           [unitless]
    Ve_i        - intial boundary layer velocity     [m/s]
    
    Outputs:  
    dydx        - expression for the momentum thickness and velocity (theta**2/Ve**6)

    Properties Used:
    N/A 
    """        
    # ADD CODE HERE !!!! 

    dydx = 0.45 * nu * getVe(x, x_i, Ve_i) ** 5

    return dydx 
    
def getVe(x,x_i,Ve_i):
    """ Interpolates the bounday layer velocity over a new dimension of x 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    x         - new x dimension                    [unitless]
    x_i       - old x dimension                    [unitless]
    Ve_i      - old boundary layer velocity values [m/s] 
    
    Outputs:  
    Ve        - new boundary layer velocity values [m/s]

    Properties Used:
    N/A 
    """
    # ADD CODE HERE !!!! 

    Ve = np.interp(x, x_i, Ve_i)

    return Ve  

def getdVe(x,x_i,dVe_i):
    """ Interpolates the derivatives of the bounday layer velocity over a new dimension of x 

    Assumptions:
    None

    Source:
    None

    Inputs: 
    x         - new x dimension                                   [unitless]
    x_i       - old x dimension                                   [unitless]
    dVe_i     - old derivative of boundary layer velocity values  [m/s-m]
    
    Outputs:  
    dVe       - new derivative of boundary layer velocity values  [m/s-m]

    Properties Used:
    N/A 
    """
    
    # ADD CODE HERE !!!! 

    dVe = np.interp(x, x_i, dVe_i)

    return dVe 

def getcf(lambda_val , Re_theta):
    """ Computes the skin friction coefficient, cf

    Assumptions:
    None

    Source:
    None

    Inputs: 
    lambda_val - thwaites separation criteria                        [unitless]
    Re_theta   - Reynolds Number as a function of momentum thickness [unitless]

    Outputs:  
    cf         - skin friction coefficient [unitless]

    Properties Used:
    N/A 
    """        
    # ADD CODE HERE !!!! 

    l = np.where(lambda_val < 0,
                 0.22 + 1.402 * lambda_val + 0.018 * lambda_val / (0.107 + lambda_val),
                 0.22 + 1.57 * lambda_val - 1.8 * lambda_val ** 2)

    cf = 2 * l / Re_theta
    
    return cf 