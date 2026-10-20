import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
# ----------------------------------------------------------------------
# heads_method.py 
# ----------------------------------------------------------------------    
def solve_heads_BL(npanel,delta_0,theta_0,delta_star_0,cf_0,H_0,Re_L,x_i,Ve_i,dVe_i,l,nu,tol= 1E0):    
    ''' Solved intergral boundary layer equations for turbulent flow  
    '''

    H1_0 = (delta_0 - delta_star_0) / theta_0
    Ve_0 = getVe(x_i[0],x_i,Ve_i)    
    y = [theta_0, H1_0 * Ve_0 * theta_0]

    x = np.linspace(x_i[0],x_i[-1],npanel)
    ReL_div_L = Re_L / l
    sol = odeint(odefcn,y,x,args=(ReL_div_L, x_i, Ve_i, dVe_i), rtol=tol, atol=tol)

    theta = sol[:,0]
    H1 = sol[:,1] / (getVe(x,x_i,Ve_i) * theta)
    H = getH(H1)
    Re_theta = ReL_div_L * theta * getVe(x,x_i,Ve_i)
    cf = getcf(Re_theta,H)
    delta_star = H * theta
    delta = H1 * theta + delta_star 
    Re_x = ReL_div_L * x * getVe(x,x_i,Ve_i)
               
    return H,delta_star,delta,cf,theta,Re_x,Re_theta  

def getH(H1):
    """ Computes the shape factor, H
    Assumptions:
    None
    Source:
    None
    Inputs: 
    H1       - mass flow shape factor [unitless]
    Outputs:  
    H        - shape factor [unitless]
    Properties Used:
    N/A
    """          

    H = np.where(H1 < 3.3, 
                 3., 
                 np.where(H1 < 5.39,
                          0.6778 + 1.1536 * (H1 - 3.3)**-0.326,
                          1.1 + 0.86 * (H1 - 3.3)**-0.777))


    return H 

def getH1(H) :    
    """ Computes the mass flow shape factor, H1
    Assumptions:
    None
    Source:
    None
    Inputs: 
    H        - shape factor [unitless]
    Outputs:  
    H1       - mass flow shape factor [unitless]
    Properties Used:
    N/A 
    """ 

    H1 = np.where(H > 1.6, 
                  3.3 + 1.5501*(H - 0.6778)**-3.064,
                  3.3 + 0.8234*(H - 1.1)**-1.287)
    
    return H1

def odefcn(y,x,ReL_div_L, x_i, Ve_i, dVe_i): 
    """ Computes boundary layer functions using SciPy ODE solver 
    Assumptions:
    None
    Source:
    None
    Inputs:  
    y           - initial conditions of functions               [unitless]
    x           - new x values at which to solve ODE            [unitless]
    ReL_div_L   - ratio of Reynolds number to length of surface [unitless]
    x_i         - intial array of x values                      [unitless]
    Ve_i        - intial boundary layer velocity                [m/s]
    dVe_i       - initial derivative of bounday layer velocity  [m/s-m]

    Outputs:  
    f           - 2D function of momentum thickness and the product of 
                  the velocity,momentum thickness and the mass flow shape factor
    Properties Used:
    N/A 
    """     

    Ve_x = getVe(x,x_i,Ve_i)
    dVe_x = getdVe(x,x_i,dVe_i)

    theta = y[0]
    H1 = y[1] / (Ve_x * theta)

    H = getH(H1)
    Re_theta = ReL_div_L * theta * Ve_x
    cf = getcf(Re_theta,H)

    dydx_1 = np.atleast_1d(0.5*cf - (theta/Ve_x)*(2 + H)*dVe_x)
    dydx_2 = Ve_x * 0.0306 * (H1 - 3)**-0.6169 
    
    return  [dydx_1[0],dydx_2] 

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

    Ve = np.interp(x, x_i, Ve_i)
    return Ve 

def getdVe(x,x_i,dVe_i):
    """ Interpolates the derivatives of the bounday layer velocity over a new dimension of x

    Assumptions:
    None
    Source:
    None
    Inputs: 
    x         - new x dimension                                  [unitless]
    x_i       - old x dimension                                  [unitless]
    dVe_i     - old derivative of boundary layer velocity values [m/s-m] 

    Outputs:  
    dVe       - new derivative of boundary layer velocity values [m/s-m]
    Properties Used:
    N/A 
    """         
    dVe = np.interp(x, x_i, dVe_i) 

    return dVe  

def getcf(Re_theta,H): 
    """ Computes the skin friction coefficient, cf
    Assumptions:
    None
    Source:
    None
    Inputs: 
    Re_theta - Reynolds Number as a function of momentum thickness [m]
    H        - shape factor                                        [unitless]
    Outputs:  
    cf       - skin friction coefficient  [unitless]
    Properties Used:
    N/A 
    """    

    cf = 0.246 * 10**(-0.678*H)*Re_theta**(-0.268)

    return cf 