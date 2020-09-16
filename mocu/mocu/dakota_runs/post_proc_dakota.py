import sys
import numpy as np
import matplotlib.pyplot as plt


def read_dakota_dat_file( dat_file ):
    
    f = open( dat_file , "r" )
    
    data   = f.readlines()
    data   = [ di.split() for di in data ]
    header = data[0]
    data   = data[1:]
    xyz    = [ di[2:] for di in data ]
    xyz    = [ [ float(dij) for dij in di ] for di in xyz ]
    
    f.close()
    
    return header , xyz
    
def main( dat_file ):
    
    header , xyz = read_dakota_dat_file( dat_file )
    
    return xyz


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

    
if __name__ == '__main__':
    
    xyz = main( sys.argv[1] )

    #plt.semilogy( [ xi[0] for xi in xyz ] , [ xi[1] for xi in xyz ] , '-o' )
    
    plt.scatter( [ xi[0] for xi in xyz ] , \
                 [ xi[1] for xi in xyz ] , \
                 c=[ xi[2] for xi in xyz ] , s=100 , cmap='coolwarm' )
    for i in range(len(xyz)-1):
        line = plt.plot( [xyz[i][0],xyz[i+1][0]] , [xyz[i][1],xyz[i+1][1]] , 'k--' , lw=1 )[0]
        add_arrow( line , size=10 )
    
    # plt.plot( [ xi[0] for xi in xyz ] , \
    #           [ xi[1] for xi in xyz ] , 'k--' , lw=1 )[0]
    # add_arrow( line , size=100 )
    
    eps = 0.5
    plt.xlim( [1-eps  , 10+eps] )
    plt.ylim( [1-eps  , 10+eps ] )
    plt.xlabel( 'n_experiment' , fontsize=16 )
    plt.ylabel( 'n_psi' , fontsize=16 )
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    
    plt.figure()
    plt.plot( [ xi[2] for xi in xyz ] , lw=2 )
    plt.xlabel( 'n_optimizer' , fontsize=16 )
    plt.ylabel( 'J' , fontsize=16 )    
    plt.tight_layout()
    
    plt.show()

    
