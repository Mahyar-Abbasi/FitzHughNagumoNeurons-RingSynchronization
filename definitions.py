import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#sol[0,0::2] ~ u and sol[0,1::2] ~ v

class FHN_sol_class:
    def __init__(self,sol):
       self.sol_array=sol
       self.dt=0.1
       self.t_range=np.arange(1000,5000+self.dt,self.dt)
       self.n=int(sol.shape[-1]/2)

    def parameters_get(self,param_str):
        self.parameters_string=param_str

    def parameters_output(self):
        return self.parameters_string
    
    def u_snapshot(self,t=None):
        if t==None:
            t=self.t_range[-1]
    
        plt.figure(figsize=(10,6))
        plt.scatter(range(1,self.n+1),self.sol_array[int((t-self.t_range[0])/self.dt),0::2],c="red")
        plt.xlabel("node index")
        plt.ylabel("u")
        plt.title(f"Snapshot at t={t:.0f}, "+self.parameters_output(),y=1.05)
        plt.ylim(-2.5,2.2)
        plt.show()

    """
    def u_plot_initial_state(self):

        plt.figure(figsize=(10,6))
        plt.scatter(range(1,self.n+1),self.sol_array[0,0::2],c="red")
        plt.xlabel("node index")
        plt.ylabel("u")
        plt.title("Initial state",y=1.05)
        plt.ylim(-2.5,2.2)
        plt.show()
    """

    def u_v_plot(self):

        u_range=np.arange(-2.1,2.1,0.01)
        plt.figure(figsize=(10,6))
        plt.plot(u_range, u_range- u_range**3/3, color="black",lw=0.5)    
        plt.plot([-self.a,-self.a],[-1.1,1.1],color="black",lw=0.5)
        plt.scatter(self.sol_array[-1,0::2],self.sol_array[-1,1::2],c="red",s=20,label=f"snapshot at t={self.t_range[-1]:.0f}")
        plt.xlabel("u")
        plt.ylabel("w")
        plt.legend()
        plt.ylim(-1.1,1.1)
        plt.xlim(-2.1,2.1)
        plt.show()


    def LOP(self,delta=None,t=None):
        if delta==None:
            delta=int(self.n/40)       

        if t==None:
            t=self.t_range[-1]

        
        ind=int((t-self.t_range[0])/self.dt)

        phase_array=np.arctan(self.sol_array[ind,1::2]/self.sol_array[ind,0::2])
        for i in range(len(phase_array)):
            if self.sol_array[ind,0::2][i]<0:
                phase_array[i]+=np.pi

        lop_list=[]

        for i in range(len(phase_array)):

            ind_list=list(np.arange(i-delta,i+1))
            ind_list=ind_list+list(np.arange(i+1,i+delta+1)%len(phase_array))
            lop_list+=[(1/(2*delta+1))*np.abs(np.sum(np.exp(1j*phase_array[ind_list])))]

        return np.array(lop_list)
    

    def LOP_matrix_show(self,delta=None,t_start=0):
        if delta==None:
            delta=int(self.n/40)
        
        times_list=np.arange(int(t_start/(self.t_range[-1]-self.t_range[-2])),len(self.t_range),100)
        lop_matrix=np.empty((len(times_list),self.n))

        for count,i in enumerate(times_list):

            phase_array=np.arctan(self.sol_array[i,1::2]/self.sol_array[i,0::2])
            for w in range(len(phase_array)):
                if self.sol_array[i,0::2][w]<0:
                    phase_array[w]+=np.pi

            lop_list=[]


            for k in range(len(phase_array)):

                ind_list=list(np.arange(k-delta,k+1))
                ind_list=ind_list+list(np.arange(k+1,k+delta+1)%len(phase_array))
                lop_list+=[(1/(2*delta+1))*np.abs(np.sum(np.exp(1j*phase_array[ind_list])))]

            lop_matrix[count,:]=np.array(lop_list)


        plt.figure(figsize=(10,8))
        im=plt.imshow(lop_matrix[::-1],cmap="inferno",aspect=lop_matrix.shape[1]/lop_matrix.shape[0])
        clb=plt.colorbar(im)
        clb.ax.set_title("Local order",fontsize=13)
        plt.title(self.parameters_output(),y=1.04,fontsize=16)
        plt.xticks([0,self.n-1],[1,self.n])
        plt.yticks([0,lop_matrix.shape[0]-1],[int(np.round(self.t_range[-1])),self.t_range[0]])
        plt.xlabel("node index",fontsize=16)
        plt.ylabel("time",fontsize=16)
        plt.tight_layout()
        plt.show()
        

    def mean_phase_velocity_calculator(self,t_i=None,t_f=None):
        if t_f==None:
            t_f=self.t_range[-1]

        if t_i==None:
            t_i=self.t_range[0]

    
        phase_vel_array=np.empty(self.n)

        for i in range(self.n):
            u_array=self.sol_array[int((t_i-self.t_range[0])/self.dt):int((t_f-self.t_range[0])/self.dt), 2*i]
            spike_counter=0
            for j in range(len(u_array)):

                if j>0 and j< len(u_array)-1:
                    if u_array[j]>u_array[j+1] and u_array[j]>u_array[j-1]:
                        spike_counter+=1

            phase_vel_array[i]=(2*np.pi*spike_counter)/(t_f-t_i)

        self.mean_phase_vel=phase_vel_array
    

    def phase_frequency_subplot(self):

        fig=plt.figure(figsize=(14,6))

        ax=fig.add_subplot(1,2,1, title=self.parameters_output(), xlabel="node index", ylabel="u")
        ax.scatter(range(1,self.n+1), self.sol_array[-1,0::2], c="red")
        ax.set_ylim(-2.5,2.2)
        plt.title(self.parameters_output(), y=1.05)

        ax=fig.add_subplot(1,2,2, xlabel="node index", ylabel="\u03C9")
        ax.scatter(range(1,self.n+1), self.mean_phase_vel ,c="black", label=f"r={self.r}, \u03C3={self.sigma}")
        ax.set_ylim(2.2,3)
        plt.title(f"Mean phase velocity", y=1.05)
        plt.legend(fontsize=14)    
        
        plt.tight_layout()
        plt.show()

#######################################

#sol[0,0::2] ~ u and sol[0,1::2] ~ v

def FHN_ring_solver(n,r,sigma,eps=0.05,phi=np.pi/2-0.1,a=0.5,t_f=5000,t_i=1000,dt=0.1,initial_state=None):
    def coupled_ring(n,k):
        a=np.zeros((n,n))
        for i in range(n):
            for t in range(1,k+1):
        #filling from left
                a[i,i-t]=1
        #filling from right
                a[i,(i+t)%n]=1
            
        return a
      
    adj_matrix=coupled_ring(n,int(r*n))
    b_matrix=np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])

    def coupled_ode(t,y,n,r,adj_matrix,eps,sigma,b_matrix,a):
        y_u=y[0::2]
        y_v=y[1::2]
        y_dot=np.empty(len(y))
        y_dot[0::2]=(1/eps)*((1-b_matrix[0,0]*sigma)*y_u-y_u**3/3-(1+b_matrix[0,1]*sigma)*y_v+
                             (b_matrix[0,0]*sigma*(adj_matrix@y_u))/(2*int(r*n))+
                             (b_matrix[0,1]*sigma*(adj_matrix@y_v))/(2*int(r*n)))
        
        y_dot[1::2]=1*((1-sigma*b_matrix[1,0])*y_u+a-sigma*b_matrix[1,1]*y_v+
                       (b_matrix[1,0]*sigma*(adj_matrix@y_u))/(2*int(r*n))+
                       (b_matrix[1,1]*sigma*(adj_matrix@y_v))/(2*int(r*n)))
                                                                                     
        return y_dot
    
    if initial_state is None: 
        random_phases=np.random.uniform(0,2*np.pi,size=(n))
        initial_state=np.empty(2*n)
        initial_state[0::2]=2*np.cos(random_phases)
        initial_state[1::2]=2*np.sin(random_phases)

    t_range=np.arange(t_i,t_f+dt,dt)

    
    sol=solve_ivp(coupled_ode, y0=initial_state, t_span=[0,t_f+dt], t_eval=t_range,
                   args=(n,r,adj_matrix,eps,sigma,b_matrix,a)).y.T

    return sol