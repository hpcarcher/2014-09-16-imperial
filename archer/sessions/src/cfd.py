#!/usr/bin/env python

"""
Computational Fluid Dynamics calculation.

Simulation of flow in a 2D square box using the Jacobi algorithm. The
box has an inlet on its top edge and an outlet on its right edge.

Written by EPCC 2014.
"""

import sys
import time
import ConfigParser

from jacobi import jacobi

def main(argv):
    """
    Read configuration file, run simulation and output stream function
    data into a file.

    Usage: cfd.py <config file> <output file> [quiet]

    If quiet flag is provided then no status information is printed.

    Keyword arguments:
    argv -- list of command-line arguments. This must have, at least,
            two elements, the configuration file name and stream
	    function data output file name.
    """

    if len(argv) < 2:
        print "Usage: cfd.py <config file> <output file> [quiet]"
        sys.exit(1)

    config_file = argv[0]
    dat_file = argv[1]
    quiet = (len(argv) == 3) and (argv[2] == "quiet")

    config_parser = ConfigParser.RawConfigParser()
    config_parser.read(config_file)
    iterations = config_parser.getint('Simulation', 'iterations')
    edge = config_parser.getint('Grid', 'edge')
    inlet_x = config_parser.getint('Inlet', 'x')
    inlet_width = config_parser.getint('Inlet', 'width')
    outlet_y = config_parser.getint('Outlet', 'y')
    outlet_height = config_parser.getint('Outlet', 'height')

    cfd(iterations, edge, inlet_x, inlet_width, outlet_y, outlet_height, dat_file, quiet)

    sys.exit(0)


def cfd(iterations, edge, inlet_x, inlet_width, outlet_y, outlet_height, dat_file, quiet=True):
    """
    Run simulation and output stream function data into a file.

    Keyword arguments:
    iterations -- number of iterations to update stream function over.
    edge -- width and height of box.
    inlet_x -- point on top edge of box where inlet begins.
    inlet_width -- inlet width.
    inlet_y -- point on right edge where outlet begins.
    inlet_height -- outlet height.
    dat_file -- stream function data output file name.
    quiet -- if True then no status information is printed.
    """

    width = edge
    height = edge

    assert inlet_x <= width
    assert inlet_x + inlet_width < width
    assert outlet_y <= height
    assert outlet_y + outlet_height < height
    
    if (not quiet):
	sys.stdout.write("\n2D CFD Simulation\n")
	sys.stdout.write("=================\n")
	sys.stdout.write("   Iterations = {0}\n".format(iterations))
	sys.stdout.write("         Edge = {0}\n".format(edge))
	sys.stdout.write("      Inlet X = {0}\n".format(inlet_x))
	sys.stdout.write("  Inlet width = {0}\n".format(inlet_width))
	sys.stdout.write("     Outlet Y = {0}\n".format(outlet_y))
	sys.stdout.write("Outlet height = {0}\n".format(outlet_height))
	sys.stdout.write("Grid size     = {0} x {1}\n".format(width, height))
    time_starts = time.time()

    # Initialise stream function, psi, to 0.
    psi = [[0 for col in range(width+2)] for row in range(height+2)]

    # Set the boundary conditions.
    set_inlet_boundary(psi, width, inlet_x, inlet_width)
    set_outlet_boundary(psi, width, outlet_y, outlet_height)

    time_ends = time.time()
    if (not quiet):
	sys.stdout.write("\nInitialisation took {0:.5f}s\n".format(time_ends-time_starts))
    
    # Call the Jacobi iterative loop (and calculate timings).
    if (not quiet):
	sys.stdout.write("\nStarting main Jacobi loop...\n")
    time_starts = time.time()

    jacobi(iterations, psi)

    time_ends = time.time()
    if (not quiet):
	sys.stdout.write("...finished\n")
	sys.stdout.write("\nCalculation took {0:.5f}s\n\n".format(time_ends-time_starts))
    
    write_data(width, height, psi, dat_file)


def set_inlet_boundary(psi, box_width, x, width):
    """
    Set the boundary conditions at the inlet on the top edge of the
    box. 

    Keyword arguments:
    psi -- 2D array of initial stream function values.
    box_width -- width of box.
    x -- point on top edge of box where inlet begins.
    width -- inlet width.
    """

    for i in range(x + 1, x + width):
        psi[0][i] = float(i - x)
    for i in range(x + width, box_width + 1):
        psi[0][i] = float(width)


def set_outlet_boundary(psi, box_width, y, height):
    """
    Set the boundary conditions at the outlet on the right edge of the
    box. 

    Keyword arguments:
    psi -- 2D array of initial stream function values.
    box_width -- width of box.
    y -- point on right edge where outlet begins.
    height -- outlet height.
    """

    for j in range(1, y + 1):
        psi[j][box_width + 1] = float(height)
    for j in range(y + 1, y + height):
        psi[j][box_width + 1] = float(height - j + y)


def write_data(width, height, psi, dat_file):
    """
    Compute flow pattern - velocity field - from stream function and
    save. 

    Keyword arguments:
    width -- box width.
    height -- box height.
    psi -- 2D array of stream function values.
    dat_file -- output file.
    """

    out = open(dat_file, "w")
    out.write("{0} {1}\n".format(width, height))
    # Loop over stream function, skipping boundaries.
    for i in range(1, height + 1):
        for j in range(1, width + 1):

            x_velocity = (psi[i][j+1] - psi[i][j-1]) / 2.0
            y_velocity = (psi[i-1][j] - psi[i+1][j]) / 2.0
            mvs = (x_velocity + y_velocity)**2
            # Scale the magnitude
            modvsq = mvs**0.3
            out.write("{0:5d} {1:5d} {2:10.5f} {3:10.5f} {4:10.5f}\n".format(i-1, j-1, x_velocity, y_velocity, modvsq))
    out.close()

if __name__ == "__main__":
    main(sys.argv[1:])
