import numpy as np    
import matplotlib.pyplot as plt 
from  solve_thwaites_BL import solve_thwaites_BL
from  solve_heads_BL    import solve_heads_BL
from  michel_criterion  import michel_criterion

def main():
    problem_2_part_a()
    problem_2_part_b()
    problem_2_part_c()
    
    return 

def problem_2_part_a():

    Re_L = 1e6
    l = 1.0
    x = np.linspace(1e-6, l, 100)
    Ve = np.ones_like(x) * 10.0 # (arbitrary) m/s
    dVe = np.zeros_like(x) # (arbitrary) m/s^2
    theta_0 = 1e-12 # m
    nu = Ve[0] * l / Re_L # m^2/s

    # Thwaites method for laminar flow 
    H_lam,delta_star_lam,delta_lam,cf_lam,theta_lam,Re_x_lam,Re_theta_lam = solve_thwaites_BL(nu,l,Re_L,x,Ve,dVe,theta_0)

    # Balsuis Solutions
    theta_blasius = 0.664 * x / np.sqrt(Re_L * x / l)
    delta_blasius = 7.83 * theta_blasius
    theta_star_blasius = 1.72 * x / np.sqrt(Re_L * x / l)
    H_blasius = theta_star_blasius / theta_blasius
    cf_blasius = 0.664 / np.sqrt(Re_L * x / l)

    fig = plt.figure()
    plt.plot(x, delta_lam, label='Thwaites Boundary Layer', linestyle='--')
    plt.plot(x, delta_star_lam, label='Thwaites Displacement Thickness', linestyle='--')
    plt.plot(x, theta_lam, label='Thwaites Momentum Thickness', linestyle='--')
    plt.plot(x, delta_blasius, label='Blasius Solution', linestyle='-')
    plt.plot(x, theta_star_blasius, label='Blasius Displacement Thickness', linestyle='-')
    plt.plot(x, theta_blasius, label='Blasius Momentum Thickness', linestyle='-')
    plt.xlabel('x (m)')
    plt.ylabel('Boundary Layer Thickness (m)')
    plt.title('Laminar Boundary Layer Thicknesses')
    plt.legend()
    fig.savefig('laminar_boundary_layer_thicnkess.png')
    fig = plt.figure()
    plt.plot(x, H_lam, label='Thwaites Shape Factor', linestyle='--')
    plt.plot(x, H_blasius, label='Blasius Shape Factor', linestyle='-')
    plt.xlabel('x (m)')
    plt.ylabel('Shape Factor H')
    plt.title('Laminar Boundary Layer Shape Factor')
    plt.legend()
    fig.savefig('laminar_boundary_layer_shape_factor.png')
    fig = plt.figure()
    plt.plot(x[1:], cf_lam[1:], label='Thwaites Skin Friction Coefficient', linestyle='--')
    plt.plot(x[1:], cf_blasius[1:], label='Blasius Skin Friction Coefficient', linestyle='-')
    plt.xlabel('x (m)')
    plt.ylabel('Local Skin Friction Coefficient')
    plt.title('Laminar Boundary Layer Skin Friction Coefficient')
    plt.legend()
    fig.savefig('laminar_skin_friction.png')

    cf_total = np.sum(cf_lam[1:] * np.diff(x)) # Integrate to get total skin friction coefficient
    print(f'Total Skin Friction Coefficient: {cf_total:.6f}')
    cf_total_blasius = np.sum(cf_blasius[1:] * np.diff(x))
    print(f'Total Skin Friction Coefficient (Blasius): {cf_total_blasius:.6f}')

    transition_index, transition_flag = michel_criterion(Re_theta_lam, Re_x_lam)
    if transition_flag:
        print(f'Transition occurs at x = {x[transition_index]:.6f} m')
    else:
        print('No transition occurs in the laminar boundary layer.')
        while (not transition_flag):
            l *= 2.
            x_search = np.linspace(1e-12, l, int(l * 1000))
            Re_x = x_search * Ve[0] / nu                        
            theta_blasius = 0.664 * x_search / np.sqrt(Re_x)      
            Re_theta = theta_blasius * Ve[0] / nu
            transition_index, transition_flag = michel_criterion(Re_theta, Re_x)
        print(f'Transition can be expected, according to blasius, to occur at x = {x_search[transition_index]:.6f} m')
    return 


def problem_2_part_b():
    
    Re_L_tur = 1e8
    l = 1.0
    Ve_tur = 1.0
    nu = Ve_tur * l / Re_L_tur
    delta_0_tur = 0.0006
    delta_star_0_tur = 0.0001
    theta_0_tur = 8e-5
    cf_0_tur = 0.0005
    H_0_tur = 2.61
    n_points = 100
    x_tur = np.linspace(1e-6, l, n_points)
    Ve_tur = np.ones_like(x_tur)
    dVe_tur = np.zeros_like(x_tur)
    

 
    # Turbulent Boundary Layer Calculation  
    H_turb,delta_star_turb,delta_turb,cf_turb,theta_turb,Re_x_turb,Re_theta_turb = solve_heads_BL(n_points,delta_0_tur,theta_0_tur,delta_star_0_tur,cf_0_tur,
                                                                                                  H_0_tur,Re_L_tur,x_tur,Ve_tur,dVe_tur, l, nu)

    fig = plt.figure()
    plt.plot(x_tur, delta_turb, label='Boundary Layer')
    plt.plot(x_tur, delta_star_turb, label='Displacement Thickness')
    plt.plot(x_tur, theta_turb, label='Momentum Thickness')
    plt.xlabel('x (m)')
    plt.ylabel('Boundary Layer Thickness (m)')
    plt.title('Turbulent Boundary Layer Thicknesses')
    plt.legend()
    fig.savefig('turbulent_boundary_layer_thicnkess.png')
    fig = plt.figure()
    plt.plot(x_tur, H_turb, label='Turbulent Shape Factor')
    plt.xlabel('x (m)')
    plt.ylabel('Shape Factor H')
    plt.title('Turbulent Boundary Layer Shape Factor')
    plt.legend()
    fig.savefig('turbulent_boundary_shape_factor.png')
    fig = plt.figure()
    plt.plot(x_tur, cf_turb, label='Turbulent Skin Friction Coefficient')
    plt.xlabel('x (m)')
    plt.ylabel('Local Skin Friction Coefficient')
    plt.title('Turbulent Boundary Layer Skin Friction Coefficient')
    plt.legend()
    fig.savefig('turbulent_skin_friction.png')

    print(f'Boundary Layer Thickness at end of plate: {delta_turb[-1]:.6f} m')
    print(f'Displacement Thickness at end of plate: {delta_star_turb[-1]:.6f} m')
    print(f'Momentum Thickness at end of plate: {theta_turb[-1]:.6f} m')
    print(f'Shape Factor at end of plate: {H_turb[-1]:.6f}')
    print(f'Skin Friction Coefficient at end of plate: {cf_turb[-1]:.6f}')

    cf_total = np.sum(cf_turb[1:] * np.diff(x_tur)) # Integrate to get total skin friction coefficient
    print(f'Total Skin Friction Coefficient: {cf_total:.6f}')
 
    
      
     # ADD CODE HERE !!!! 
    
def problem_2_part_c():  

    Re_L = 1e7
    l = 4.0
    Ve = np.ones(100) # (arbitrary) m/s
    dVe = np.zeros(100) # (arbitrary) m/s^2
    x = np.linspace(0, l, 100)
    nu = Ve[0] * l / Re_L # m^2/s
    theta_0 = 1e-12 # m 
  
    # Thwaites method for laminar flow 
    H_lam,delta_star_lam,delta_lam,cf_lam,theta_lam,Re_x_lam,Re_theta_lam = solve_thwaites_BL(nu,l,Re_L,x,Ve,dVe,theta_0) 
 
    # Find transition point 
    transition_index,transition_flag = michel_criterion(Re_theta_lam,Re_x_lam)
    
    if transition_flag:
        n_points = len(x[transition_index:]) # number of points for turbulent boundary layer calculation
        delta_0_tur = delta_lam[transition_index]
        theta_0_tur = theta_lam[transition_index]
        delta_star_0_tur = delta_star_lam[transition_index]
        cf_0_tur = cf_lam[transition_index]
        H_0_tur = H_lam[transition_index]
        Re_L_tur = Re_L
        x_tur = x[transition_index:]
        Ve_tur = Ve[transition_index:]
        dVe_tur = dVe[transition_index:]

        # Turbulent Boundary Layer Calculation  
        H_turb,delta_star_turb,delta_turb,cf_turb,theta_turb,Re_x_turb,Re_theta_turb = solve_heads_BL(n_points,delta_0_tur,theta_0_tur,delta_star_0_tur,cf_0_tur,
                                                                                                      H_0_tur,Re_L_tur,x_tur,Ve_tur,dVe_tur, l, nu) 

        # Concatenate results
        
        delta_star = np.concatenate((delta_star_lam[:transition_index], delta_star_turb))
        delta = np.concatenate((delta_lam[:transition_index], delta_turb))
        cf = np.concatenate((cf_lam[:transition_index], cf_turb))
        theta = np.concatenate((theta_lam[:transition_index], theta_turb))
        Re_x = np.concatenate((Re_x_lam[:transition_index], Re_x_turb))
        Re_theta = np.concatenate((Re_theta_lam[:transition_index], Re_theta_turb))
        H = np.concatenate((H_lam[:transition_index], H_turb))

        print(f'Transition occurs at x = {x[transition_index]:.6f} m')


    else: # no transition 
        
        delta_star = delta_star_lam
        delta      = delta_lam
        cf         = cf_lam
        theta      = theta_lam 
        Re_x       = Re_x_lam 
        Re_theta   = Re_theta_lam  
        H          = H_lam 

    
    fig = plt.figure()
    plt.plot(x, delta, label='Boundary Layer')
    plt.plot(x, delta_star, label='Displacement Thickness')
    plt.plot(x, theta, label='Momentum Thickness')
    plt.xlabel('x (m)')
    plt.ylabel('Boundary Layer Thickness (m)')
    plt.title('Boundary Layer Thicknesses')
    plt.legend()
    fig.savefig('combined_boundary_layer_thicnkess.png')
    fig = plt.figure()
    plt.plot(x, H, label='Shape Factor')
    plt.xlabel('x (m)')
    plt.ylabel('Shape Factor H')
    plt.title('Boundary Layer Shape Factor')
    plt.legend()
    fig.savefig('combined_boundary_shape_factor.png')
    fig = plt.figure()
    plt.plot(x[1:], cf[1:], label='Skin Friction Coefficient')
    plt.xlabel('x (m)')
    plt.ylabel('Local Skin Friction Coefficient')
    plt.title('Boundary Layer Skin Friction Coefficient')
    plt.legend()
    fig.savefig('combined_skin_friction.png')

    cf_total = np.sum(cf[1:] * np.diff(x)) / l # Integrate to get total skin friction coefficient
    print(f'Total Skin Friction Coefficient: {cf_total:.6f}')
     
    return 
  
if __name__ == '__main__':
    main()
    plt.show()
 