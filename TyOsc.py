import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path[:0] = ['..']
import gillespy
import time
import itertools

class Tyson2StateOscillator(gillespy.Model):
    """
    Here, as a test case, we run a simple two-state oscillator (Novak & Tyson 
    2008) as an example of a stochastic reaction system.
    """
    def __init__(self, parameter_values=None):
        """
        """
        system_volume = 300 #system volume
        gillespy.Model.__init__(self, name="tyson-2-state", volume=system_volume)
        self.timespan(np.linspace(0,100,101))
        # =============================================
        # Define model species, initial values, parameters, and volume
        # =============================================    
        
        # Parameter values  for this biochemical system are given in 
        # concentration units. However, stochastic systems must use population
        # values. For example, a concentration unit of 0.5mol/(L*s)
        # is multiplied by a volume unit, to get a population/s rate
        # constant. Thus, for our non-mass action reactions, we include the 
        # parameter "vol" in order to convert population units to concentration
        # units. Volume here = 300.

        P = gillespy.Parameter(name='P', expression=parameter_values[0])
        kt = gillespy.Parameter(name='kt', expression=parameter_values[1])
        kd = gillespy.Parameter(name='kd', expression=parameter_values[2])
        a0 = gillespy.Parameter(name='a0', expression=parameter_values[3])
        a1 = gillespy.Parameter(name='a1', expression=parameter_values[4])
        a2 = gillespy.Parameter(name='a2', expression=parameter_values[5])
        kdx = gillespy.Parameter(name='kdx', expression=parameter_values[6])
        self.add_parameter([P, kt, kd, a0, a1, a2, kdx])
        
        # Species
        # Initial values of each species (concentration converted to pop.)
        X = gillespy.Species(name='X', initial_value=int(0.65609071*system_volume))
        Y = gillespy.Species(name='Y', initial_value=int(0.85088331*system_volume))
        self.add_species([X, Y])
        
        # =============================================  
        # Define the reactions within the model
        # =============================================  
        
        # creation of X:
        rxn1 = gillespy.Reaction(name = 'X production',
                        reactants = {},
                        products = {X:1},
                        propensity_function = 'vol*1/(1+(Y*Y/((vol*vol))))')
        
        # degradadation of X:
        rxn2 = gillespy.Reaction(name = 'X degradation',
                    reactants = {X:1},
                    products = {},
                    rate = kdx)
        
        # creation of Y:
        rxn3 = gillespy.Reaction(name = 'Y production',
                    reactants = {X:1},
                    products = {X:1, Y:1},
                    rate = kt)
        
        # degradation of Y:
        rxn4 = gillespy.Reaction(name = 'Y degradation',
                    reactants = {Y:1},
                    products = {},
                    rate = kd)
            
        # nonlinear Y term:
        rxn5 = gillespy.Reaction(name = 'Y nonlin',
                    reactants = {Y:1},
                    products = {},
                    propensity_function = 'Y/(a0 + a1*(Y/vol)+a2*Y*Y/(vol*vol))')
        
        self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5])



if __name__ == '__main__':
    
    #varying all parameters
    """  
    comb= [
        np.linspace(2*0.7,2*1.3,5),
        np.linspace(20*0.7,20*1.3,5),
        np.linspace(1*0.7,1*1.3,5),
        np.linspace(0.005*0.7,0.005*1.3,5),
        np.linspace(0.05*0.7,0.05*1.3,5),
        np.linspace(0.1*0.7,0.1*1.3,5),
        np.linspace(1*0.7,1*1.3,5),
        ]
    comblist=[]
    for combination in itertools.product(*comb):
        comblist.append(combination)
        """
        
        
    #use to generate training data, varying last three parameters
    """
    
    comb= [
        np.linspace(0.05*0.7,0.05*1.3,5),
        np.linspace(0.1*0.7,0.1*1.3,5),
        np.linspace(1*0.7,1*1.3,5),
        ]
    comblist=[]
    for combination in itertools.product(*comb):
        vec=[2,20,1,0.005,combination[0],combination[1],combination[2]]
        comblist.append(vec)
        """
    
    #use to generate test data, varying last three parameters with different increment
    
    comb= [
        np.linspace(0.05*0.7,0.05*1.3,4),
        np.linspace(0.1*0.7,0.1*1.3,4),
        np.linspace(1*0.7,1*1.3,4),
        ]
    comblist=[]
    for combination in itertools.product(*comb):
        vec=[2,20,1,0.005,combination[0],combination[1],combination[2]]
        comblist.append(vec)
        
        
    
    """
    Pvals=np.linspace(2*0.7,2*1.3,5)
    ktvals=np.linspace(20*0.7,20*1.3,5)
    kdvals=np.linspace(1*0.7,1*1.3,5)
    a0vals=np.linspace(0.005*0.7,0.005*1.3,5)
    a1vals=np.linspace(0.05*0.7,0.05*1.3,5)
    a2vals=np.linspace(0.1*0.7,0.1*1.3,5)
    kdxvals=np.linspace(1*0.7,1*1.3,5)
    """
    
    #t1=time.time()
    
    
    
    
    dataY=[]
    data=[]
    for k in range(0,len(comblist)):
        tyson_model=Tyson2StateOscillator(parameter_values=comblist[k])
        tyson_trajectories = tyson_model.run(show_labels=False)
        data.append(comblist[k])
        dataY.append(tyson_trajectories[0][:,1])
        
    dataY=np.array(dataY)
    data=np.array(data)
       
    
    #ttot=time.time()-t1
    
    #pickle save training data
    """
    
    import cPickle as pickle
    pickle.dump( data, open( "savedata.p", "wb" ) )
    pickle.dump( dataY, open( "savedataY.p", "wb" ) )
    """
    
    #pickle save test data
    import cPickle as pickle
    pickle.dump( data, open( "savedatatest.p", "wb" ) )
    pickle.dump( dataY, open( "savedataYtest.p", "wb" ) )
    
    
    #pickle load
    #import cPickle as pickle
    #data = pickle.load( open( "save.p", "rb" ) )
    
    
    # =============================================  
    # plot just the first trajectory, 0, in both time and phase space:
    # ============================================= 
    """
    from matplotlib import gridspec
    
    gs = gridspec.GridSpec(1,2)
    
    
    ax0 = plt.subplot(gs[0,0])
    ax0.plot(tyson_trajectories[0][:,0], tyson_trajectories[0][:,1], 
             label='X')
    ax0.plot(tyson_trajectories[0][:,0], tyson_trajectories[0][:,2], 
             label='Y')
    ax0.legend()
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Species Count')
    ax0.set_title('Time Series Oscillation')
    
    ax1 = plt.subplot(gs[0,1])
    ax1.plot(tyson_trajectories[0][:,1], tyson_trajectories[0][:,2], 'k')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Phase-Space Plot')
    
    plt.tight_layout()
    plt.show()
    """
    
    



#test

#print tyson_trajectories
#tyson_trajectories[0][19999][3] #first array, second row, third column
#tyson_trajectories.append(5) #adds an element at the end of the list, doesnt have to be same type
