"""
Computational Fluid Dynamics JACOBI functions.

Written by EPCC 2014.
"""

def jacobi(num_iterations, psi):
    """
    Update stream function across a number of iterations.

    Keyword arguments:
    num_iterations -- number of iterations.
    psi -- 2D array of initial stream function values.
    """
    
    # Get the box dimensions.
    height = len(psi) - 2
    width = len(psi[0]) - 2

    for iteration in range(num_iterations):
	update_stream_function(width, height, psi)

def update_stream_function(width, height, psi):
    """
    Update stream function over a single iteration.

    Keyword arguments:
    width -- box width.
    height -- box height.
    psi -- 2D array of stream function values.
    """
    
    tmp = [[0 for col in range(width+2)] for row in range(height+2)]
    # Calculate updated stream function into tmp.
    for i in range(1, height+1):
        for j in range(1, width+1):
	    tmp[i][j] = 0.25 * (psi[i+1][j]+psi[i-1][j]+psi[i][j+1]+psi[i][j-1])

    # Update stream function using tmp.
    for i in range(1, height+1):
        for j in range(1, width+1):
	    psi[i][j] = tmp[i][j]
