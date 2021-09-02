import numpy as np
import numba

@numba.njit(nogil=True,fastmath=True,cache=True)
def logistic_cdf(x,mu,s):
    return 1/(1+np.exp(-(x-mu)/s))

@numba.njit(nogil=True,fastmath=True,cache=True)
def phaseIndicator(t,t0,a,b,s):
    """
    phase indicator function with logistic distribution
    ref: https://en.wikipedia.org/wiki/Logistic_distribution
    Input:
        t:  normalized phase [0-1]
        t0: phase offset [0-1]
        a:  mean [0-1]
        b:  mean [0-1]
        s:  variance [0-1]
    returns:
        phase indicator
    """
    t_ = ((t - t0)%1)*2.-1. # map [0,1]->[-1,1]
    mu_a = a*2.-1.
    # cumulative distribution function
    cdf_a = (logistic_cdf(t_,mu_a,s)- logistic_cdf(-1,mu_a,s))/(logistic_cdf(1,mu_a,s) - logistic_cdf(-1,mu_a,s))
    mu_b = b*2.-1.
    # complementary cumulative distribution function
    ccdf_b = 1-(logistic_cdf(t_,mu_b,s)- logistic_cdf(-1,mu_b,s))/(logistic_cdf(1,mu_b,s) - logistic_cdf(-1,mu_b,s))
    return cdf_a*ccdf_b

@numba.experimental.jitclass([
    ("a", numba.float32),
    ("b", numba.float32),
    ("s", numba.float32),
    ("t0", numba.float32),
    ("t1", numba.float32),
])
class phaseIndicatorPair:
    def __init__(self,a,b,s,t0,t1):
        """ create a parir of phase indicator
        Args:
            a # separation point for increasing from 0 -> 1
            b # separation point for decreasing from 1 -> 0
            s # sigma, shared variance
            t0 # normlaized phase offset [0-1] for the 1st value in the pair
            t1 # normlaized phase offset [0-1] for the 2nd value in the pair 
        """
        
        self.a = a # separation point for increasing from 0 -> 1
        self.b = b # separation point for decreasing from 1 -> 0
        self.s = s # variance
        self.t0 = t0 # normlaized phase offset [0-1] for the 1st value in the pair
        self.t1 = t1 # normlaized phase offset [0-1] for the 2nd value in the pair
    def get(self,t):
        """ return a pair of phase indicator values give noramlized time t"""
        return phaseIndicator(t,self.t0,self.a,self.b,self.s),phaseIndicator(t,self.t1,self.a,self.b,self.s)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    import argparse
    parser = argparse.ArgumentParser(description='phase indication demo')
    
    parser.add_argument('-a', type = float, default= 0.3,help="lower bound")
    parser.add_argument('-b', type = float, default= 0.7,help="upper bound")
    parser.add_argument('-s', type = float, default= 0.05,help="shared variance")
    parser.add_argument('-t0', type = float, default= 0.,help="phase offset 0")
    parser.add_argument('-t1', type = float, default= 0.5,help="phase offset 1")

    args = parser.parse_args()

    t = np.linspace(0,8,1000)
    a = args.a
    b = args.b
    s = args.s

    t0 = args.t0
    t1 = args.t1
    
    # a = 0.3
    # b = 0.7
    # s = 0.05

    # t0 = 0.
    # t1 = 0.5

    # e_i0 = phaseIndicator(t,t0,a,b,s)
    # e_i1 = phaseIndicator(t,t1,a,b,s)

    phase_indicator_pair = phaseIndicatorPair(a,b,s,t0,t1)
    e_i0,e_i1 = phase_indicator_pair.get(t)

    fig, ax= plt.subplots(1,1 ,figsize=(8,2),dpi=75)
    ax.plot(t,e_i0,label=f"t0={t0}")
    ax.plot(t,e_i1,label=f"t1={t1}")
    ax.plot([a,a],[0,1],'--',label=f'a={a}')
    ax.plot([b,b],[0,1],'--',label=f'b={b}')
    plt.legend(loc='best', frameon=False)
    plt.show()