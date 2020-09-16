from mocu.utils.toysystems import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_rhs_full_system(a,b,k,c,lam,psi,theta):
    
    def rhs_full_system(y,t):
        C      = c(a,b,k,y[0],psi,theta)
        y1_dot = lam[0] * (y[0] - 1)
        y2_dot = lam[1] * (y[1] - C) * (y[1] - a) * (y[1] - b)
        return [y1_dot , y2_dot]

    return rhs_full_system

def plot_points_in_full_space(xy,c,colors):
    a = 0; b=1; k = 5
    y1_lin = np.linspace( 0, 1, 100 )
    plt.scatter(xy[0] , xy[1] , c=colors , cmap=plt.cm.coolwarm)
    plt.plot( y1_lin , c(a,b,k,y1_lin) , 'k')

def plot_different_thetas():
    
    a = 0; b=1; 
    theta = [0.45 , 0.5 , 0.55]

    nsamp = 100
    y0 = np.linspace(0.5 - (b-a)/20 , 0.5 + (b-a)/20 , nsamp)
    tf = 30
    dt = 0.05
    t  = np.arange(0,tf,dt)
    colors = cm.coolwarm( np.linspace(0,1,len(y0)) )
    yfinal = []
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    
    for (i,th) in enumerate(theta):
        f = make_f( a , b , th )
        g = make_noise_function(0.03)
        for (y0_i,c_i) in zip(y0,colors):
            Y = np.squeeze( sdeint.itoint(f,g,y0_i,t) )
            yfinal.append(Y[-1])
            axes[i].plot(t , [y for y in Y] , c=c_i)
        axes[i].set_title(r'Boundary = ' + str(th) , fontsize=20)
    axes[0].set_xlabel(r'$t$' , fontsize=16)
    axes[0].set_ylabel(r'$c(t,\theta)$' , fontsize=16)
    
    plt.tight_layout()
    plt.show()

def f_input_output_sde( psi,theta,x0 ):
    
    tf = 30
    dt = 0.05
    t  = np.arange(0,tf,dt)

    a = 0; b=1;
    c = 0.04 * np.abs( psi-theta ) + 0.48
    
    f = make_f( a , b , c )
    g = make_noise_function( 0.03 )

    y = np.squeeze( sdeint.itoint(f,g,x0,t) )

    return y

def f_input_output_ode( psi,theta,x0 ):

    dt      = 0.05
    tf      = 30
    t       = np.arange(0,tf,dt)
    lam     = [-0.01,-1]
    k       = 5
    a = 0; b=1;
    
    c      = lambda a,b,k,y1,psi,theta : (0.48 + 0.04*np.abs(psi-theta) ) + 0.04*np.abs(b-a)*np.sin(2*np.pi*k*y1)
    f_real = make_rhs_full_system(a,b,k,c,lam,psi,theta)
    
    y      = np.squeeze( odeint( f_real , x0 , t ) )
    #y      = [ yi[1] for yi in y ]
    
    return y
    
def estimate_transition_probabilities( f_input_output , psi , theta , y0 ):

    colors = cm.coolwarm( np.linspace(0,1,len(y0)) )
    yfinal = []
    #plt.figure()
    #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    for (y0_i,c_i) in zip(y0,colors):
        Y  = f_input_output( psi , theta , y0_i )
        if (np.size(Y[0]) > 1):
            Y = [ yi[1] for yi in Y ]
        yfinal.append(Y[-1])
        #axes[2].plot(t , [y for y in Y] , c=c_i)

    # Only use y-coordinate of "real" system
    if (np.size(y0[0]) > 1):
        y0 = np.array( [ yi[1] for yi in y0 ] )

    # Estimate IC->final phase probabilities
    idx0_0 = np.where( y0 <  0.5 )[0]
    idx0_1 = np.where( y0 >= 0.5 )[0]
    yfinal = np.array( yfinal )
    n_00   = np.sum( yfinal[idx0_0] <  0.5 )
    n_01   = np.sum( yfinal[idx0_0] >= 0.5 )
    n_10   = np.sum( yfinal[idx0_1] <  0.5 )
    n_11   = np.sum( yfinal[idx0_1] >= 0.5 )
    n_0    = np.sum( yfinal <  0.5 )
    n_1    = np.sum( yfinal >= 0.5 )

    rho_ic_to_final = np.array([ [n_00/(n_00+n_01) , n_01/(n_00+n_01) ] ,
                                 [n_10/(n_10+n_11) , n_11/(n_10+n_11)] ])
    print( 'rho( final phase | ic ): ' )
    print( rho_ic_to_final )

def plot_real_system( psi , theta , y0_2 ):
    
    dt      = 0.05
    tf      = 30
    t       = np.arange(0,tf,dt)
    
    #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    colors    = cm.coolwarm( np.linspace(0,1,len(y0_2)) )
    for (y0_i,c_i) in zip(y0_2,colors):
        
        y = f_input_output_ode( psi,theta,y0_i )
        
        plt.figure(2)
        plt.plot( t , [yi[1] for yi in y] , c=c_i )
        plt.xlabel(r'$t$',fontsize=20)
        plt.ylabel(r'$c_2$',fontsize=20)
        
        plt.figure(3)
        plt.plot( [yi[0] for yi in y] , [yi[1] for yi in y] , c=c_i )
        plt.xlabel(r'$c_1$',fontsize=20)
        plt.ylabel(r'$c_2$',fontsize=20)

    c1 = np.linspace( 0 , 1 , 100 )
    C  = (0.48 + 0.04 * np.abs(psi-theta) ) + 0.04 * np.sin(2*np.pi*5*c1)
    plt.plot( c1 , C , 'k--' , lw=3 )
        
    plt.tight_layout()
    plt.show()



def main():
    
    nsamp  = 1000
    y0     = np.random.uniform( 0 , 1 , nsamp )
    y1     = np.linspace(0.45 , 0.55 , nsamp)
    y      = tuple( zip( y0 , y1 ) )
    psi    = 0.0; theta = 0.5
    
    estimate_transition_probabilities( f_input_output_sde , psi , theta , y1 )
    estimate_transition_probabilities( f_input_output_ode , psi , theta , y )

    #plot_different_thetas()
    #plot_real_system( psi , theta , y )

    
if __name__ == '__main__':
    main()

    
