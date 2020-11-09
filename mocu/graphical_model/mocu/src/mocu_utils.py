import numpy as np

def compute_quantiles( theta , rho_theta ):
    
    t_mean      = np.sum( rho_theta * theta )
    t_cumdist   = np.cumsum( rho_theta ) / np.sum( rho_theta )
    
    t_68        = np.where( (t_cumdist>=.16) & (t_cumdist<=.68+.16) )[0]
    
    if ( len(t_68) > 1 ):
        t_68        = [ t_mean-t_68[0] , t_68[-1]-t_mean ]
    else:
        t_68        = [ 0 , 0 ]
        
    t_95        = np.where( (t_cumdist>=.025) & (t_cumdist<=.95+.025) )[0]
    if ( len(t_95) > 1 ):
        t_95        = [ t_mean-t_95[0] , t_95[-1]-t_mean ]
    else:
        t_95        = [ 0 , 0 ]
    
    return t_mean , t_68 , t_95


class MOCU_data( ):
    
    def __init__( self ):
        
        self.X_chosen             = []
        self.J_optimal            = []
        self.psi_ibr              = []
        self.psi_ibr_idx          = []
        self.experimental_outcome = []
        self.MAP_PROB             = []
        self.MAP_IDX              = []
        self.MAP                  = []
        self.J_psi_theta          = []
        self.rho_theta            = []
        self.theta                = []
        self.th_mean              = []
        self.th_var               = []
        self.t_68                 = []
        self.t_95                 = []

    def set_data_with_current_metrics( self , mocu_out , idx_outcome , Theta ):
        
        psi_ibr_star       = mocu_out['psi_ibr'][mocu_out['idx_xstar'][0]][idx_outcome]
        psi_ibr_idx_star   = mocu_out['psi_ibr_idx'][mocu_out['idx_xstar'][0]][idx_outcome]
        Theta['rho_theta'] = mocu_out['rho_theta_given_x_y'][mocu_out['idx_xstar'][0]][:,idx_outcome]
        map_idx            = np.argmax( Theta['rho_theta'] )
        mapi               = Theta['theta'][map_idx]
        map_prob           = Theta['rho_theta'][map_idx]
        
        self.X_chosen.append( mocu_out['x_star'] )
        self.J_optimal.append( mocu_out['avg_omega'][mocu_out['idx_xstar'][0]] )
        self.psi_ibr.append( mocu_out['psi_ibr'][mocu_out['idx_xstar'][0]] )
        self.psi_ibr_idx.append( psi_ibr_star )

        self.experimental_outcome.append( idx_outcome )
        self.MAP_PROB.append( map_prob )
        self.MAP_IDX.append( map_idx )
        self.MAP.append( mapi )
        self.rho_theta.append( Theta['rho_theta'] )
        self.theta       = Theta['theta']
        self.J_psi_theta = mocu_out['J_psi_theta']
        
        t_idx = np.arange( len( Theta['theta'] ) )
        
        t_mean , t_68 , t_95 = compute_quantiles( t_idx , Theta['rho_theta'] )
        
        self.th_mean.append( t_mean )
        self.th_var.append( np.sum( Theta['rho_theta'] * ( t_idx - t_mean )**2 ) )
        self.t_68.append( t_68 )
        self.t_95.append( t_95 )
        
        
def print_current_mocu_metrics( mocu_out , map_idx , mapi , map_prob ):
    
    print( "CHOSEN EXPERIMENT: X_" + str(mocu_out['idx_xstar'][0]+1) + ' : ' + str(mocu_out['x_star']) )
    print( 'avg_omega_xstar = ' + str(mocu_out['avg_omega'][mocu_out['idx_xstar'][0]]) )
    print( 'psi_ibr_xstar = ' + str(mocu_out['psi_ibr'][mocu_out['idx_xstar'][0]] ) )
    print( 'rho_theta_given_xstar_y = \n' + str(mocu_out['rho_theta_given_x_y'][mocu_out['idx_xstar'][0]] ) )
    print( 'Psi(Theta | x,y) : \n' + str(mocu_out['psi_ibr']) )
    print( 'Avg_omega : \n' + str(mocu_out['avg_omega']) )
    print( 'MAP(theta) : ' + str(map_idx) + ' , ' + str(mapi) + ' , prob = ' + str(map_prob) )
    print( '\n' )
