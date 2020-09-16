import numpy as np

def cost_1parameter_example(theta,psi):
    """
    Example cost function where theta is one parameter and psi is one parameter.
    
    Inputs

    -------
    theta : numpy.ndarray(n), or numpy.ndarray([n_grid,n_grid])
    psi : numpy.ndarray(n), or numpy.ndarray([n_grid,n_grid])

    Outputs

    --------
    c : numpy.ndarray(n), or numpy.ndarray([n_grid, n_grid])
        cost values from each pair of (theta,psi)
    """
    c = psi**2 + 2*theta*psi + 10
    if (len(np.shape(psi)) == 2):
        c = [[np.maximum(el,0) for el in arr_i] for arr_i in c]
    else:
        try:
            c = [np.maximum(el,0) for el in c]
        except:
            c = np.maximum(c,0)
    c = np.array(c)
    return c


def cost_2parameter_example(theta,psi):
    """
    Example cost function where theta is two parameters and psi is one parameter.
    
    Inputs

    -------
    theta : list(n) where each element is numpy.array(2)
    psi : list(n)

    Outputs

    --------
    c : numpy.ndarray(n)
    """
    try:
        n_theta = np.shape(theta)[0]
    except:
        n_theta = np.size(theta)
    try:
        n_psi   = np.shape(psi)[0]
    except:
        n_psi   = np.size(psi)
    
    if ( (n_theta > 1) & (n_psi > 1) ):
        c = [psi_i**2 + 2*th_i[0]*psi_i + 10 for (th_i,psi_i) in zip(theta,psi)]
    elif ( (n_theta > 1) & (n_psi == 1) ):
        c = [psi**2 + 2*th_i[0]*psi + 10 for th_i in theta]
    elif ( (n_theta == 1) & (n_psi > 1) ):
        c = [psi_i**2 + 2*theta[0]*psi_i + 10 for psi_i in psi]
    else:
        # theta and psi each have only one value
        c = psi**2 + 2*theta*psi + 10
        
    try:
        c = [np.maximum(el,0) for el in c]
    except:
        c = np.maximum(c,0)
    c = np.array(c)
    return c
