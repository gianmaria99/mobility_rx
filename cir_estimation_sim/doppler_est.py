import sys
import os
sys.path.insert(0,os.getcwd())
from cir_estimation_sim.channel_model import ChannelModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import utils

class DopplerEst(ChannelModel):

    def __init__(self, l:float, n_static:list, snr:list, interval:list):
        """
            l: wavelength [m],
            n_static: list of numbers of available static paths,
            snr: list of SNR values [dB],
            interval: list of interval values [ms].
        """
        super().__init__(l=l, n_static=n_static[-1], snr=snr[-1]) 
        self.snr_list = snr
        self.n_static_list = n_static
        self.interval_list = interval

    def get_phases(self, h, init=False, from_index= True, plot=False):
        """
            Returns cir phases [LoS,t,s1,...,sn_static].
            h: cir,
            from_index: if True use the correct index to locate peaks, else performs peak detection,
            plot: wether to plot the selected paths from the given cir.
        """
        if from_index:
            ind = np.floor(self.paths['delay']*self.B).astype(int) # from paths delay
        else:
            ind = np.argsort(np.abs(h))[-len(self.paths['delay']):] # from cir peaks
        phases = np.angle(h[ind])
        if init:
            self.phases = np.zeros((self.n_static+2,2))
        self.phases[:,0] = self.phases[:,1]
        self.phases[:,1] = phases
        if plot:
            t = np.zeros(len(h))
            t[ind] = np.abs(h[ind])
            plt.stem(np.abs(self.cir), label='real')
            plt.stem(t, markerfmt='gD', label='selected paths')
            plt.grid()
            plt.legend()
            plt.show()
        return phases 
    
    def solve_system(self, phases, zetas):
        """
            Solves the system for the received input parameters. 
            phases: measured phases,
            zetas: angle of arrivals.

            returns the unknowns of the system.
        """       
        alpha = phases[2]*(np.cos(zetas[1])-1)-(phases[1]*(np.cos(zetas[2])-1))
        beta = phases[1]*np.sin(zetas[2])-(phases[2]*np.sin(zetas[1]))
        if abs(beta)<1e-5:
            eta = np.pi/2 - np.arctan(beta/alpha)
        else:
            eta = np.arctan(alpha/beta)
        A = (np.sin(zetas[1])*(1-np.cos(zetas[2]))) + (np.sin(zetas[2])*(np.cos(zetas[1])-1))
        if not (beta)/A>0:
            eta = eta + np.pi
        if eta<0:
            eta = eta + (2*np.pi)
        f_d = (phases[0]-((phases[1]*(np.cos(zetas[0]-eta)-np.cos(eta)))/(np.cos(zetas[1]-eta)-np.cos(eta))))/(2*np.pi*self.T)
        
        v = self.l/(2*np.pi*self.T)*phases[1]/(np.cos(zetas[1]-eta)-np.cos(eta))
        return eta, f_d, v 
    
    def system(self, x, phases, n_zetas):
        """
            Defines the system to solve it using the non linear least-square method, for at least 4 measured phases (i.e. 2 static paths).
            x = [f_D, v, eta]: system unknowns,
            phases: measured phases,
            zetas: angle of arrivals.

            returns an array representing the system.
        """
        results = []
        #target path
        results.append(phases[0]-(2*np.pi*self.T*(x[0]+(x[1]/self.l*(np.cos(n_zetas[0]-x[2])-np.cos(x[2]))))))
        # loop for each static path, i.e., excluding LoS and Target
        for i in range(1,len(phases)):
            results.append(phases[i]-(2*np.pi*self.T*(x[1]/self.l*(np.cos(n_zetas[i]-x[2])-np.cos(x[2])))))
        return np.array(results)
    
    def check_initial_values(self, x0):
        """
            Check if the selected initial values are within the known intervals and change them to default value if not.
            x0: [f_D(0), v(0), eta(0)]: initial values.
        """
        if x0[0]<-self.fd_max or x0[0]>self.fd_max:
            x0[0]=self.fd_max/2
        if x0[1]<0 or x0[1]>self.vmax:
            x0[1]=2
        x0[2] = x0[2]%(2*np.pi)
        return x0  
    
    def my_mod_2pi(self, phases):
        """
            Maps an array or a matrix of angles [rad] in [-pi,pi].
            phases: measured phases.

            returns phases mapped in [-pi,pi].
        """
        if phases.ndim==2:
            for j in range(phases.shape[1]):
                for i,p in enumerate(phases[:,j]):
                    while p <-np.pi:
                        p = p+2*np.pi
                    while p>np.pi:
                        p = p-2*np.pi
                    phases[i,j]=p
        elif phases.ndim==1:
            for i,p in enumerate(phases):
                    while p <-np.pi:
                        p = p+2*np.pi
                    while p>np.pi:
                        p = p-2*np.pi
                    phases[i]=p
        else:
            raise Exception("phases number of dimensions must be <= 2.")
        return phases
    
    def get_phase_diff(self, interval):
        """
            computes phase differences and AoAs, and averages them across interval.

            interval: coherence interval length in [ms].

            returns mean phase differences and AoAs.
        """
        interval = int(interval*1e-3/self.T)
        phase_diff = []
        self.k = 0
        h = self.get_cir_est(init=True)
        self.get_phases(h, init=True)
        for p in range(1,len(self.phases[:,1])):
                self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
        AoA = []#[self.paths['AoA'][1:] + np.random.normal(0,self.AoAstd,self.n_static+1)]
        for i in range(1,interval):
            self.k = i
            h = self.get_cir_est(init=False)
            self.get_phases(h,plot=False)
            ### remove LoS from other paths ###
            for p in range(1,len(self.phases[:,1])):
                self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
            ### phase difference ###
            diff = self.phases[:,1] - self.phases[:,0]
            phase_diff.append(diff)
            ### collect noisy AoA measurements ###
            AoA.append(self.paths['AoA'][1:] + np.random.normal(0,self.AoAstd,self.n_static+1))

        phase_diff = np.stack(phase_diff, axis=0)
        phase_diff = self.my_mod_2pi(phase_diff)
        AoA = np.stack(AoA,axis=0)
        return phase_diff, AoA

    def simulation(self, path:str, N:int, aoa:list=[5,3,1], save=True, relative=True, save_all=False):
        """
            perform the simulation varying all the available parameters.
            
            path: path to save the results,
            N: number of iterations per set of parameters,
            aoa: AoA std values in degrees,
            save: wether to save the results,
            relative: wether to monitor normalized results (otherwhise the absolute value will be saved),
            save_all: wether to save fD, eta and speed errors (otherwise just fD errors will be saved).

            returns the array tot_fd_error_rel with shape (N, len(self.snr_list), len(aoa), len(self.interval_list), len(self.n_static_list)).
        """
        tot_fd_error_rel = []
        tot_eta_error_rel = []
        tot_v_error_rel = []
        tot_fd_error_abs = []
        tot_eta_error_abs = []
        tot_v_error_abs = []
        
        nls_time = []
        for j in range(N):
            print("iteration: ", j, end="\r")
            fd_error_rel = []
            eta_error_rel = []
            v_error_rel = []
            fd_error_abs = []
            eta_error_abs = []
            v_error_abs = []
            self.get_positions(self.x_max,self.y_max,plot=False)
            for s in self.snr_list:
                self.SNR = s
                a_fd_err_rel = []
                a_eta_err_rel = []
                a_v_err_rel = []
                a_fd_err_abs = []
                a_eta_err_abs = []
                a_v_err_abs = []
                for a in aoa:
                    self.AoAstd = np.deg2rad(a)
                    phase_diff, AoA = self.get_phase_diff(self.interval_list[0])
                    int_fd_err_rel = []
                    int_eta_err_rel = []
                    int_v_err_rel = []
                    int_fd_err_abs = []
                    int_eta_err_abs = []
                    int_v_err_abs = []
                    for inter in self.interval_list: # interval in decreasing order
                        inter = int(inter*1e-3/self.T)
                        AoA = AoA[:inter]
                        phase_diff = phase_diff[:inter]
                        ### time average ###
                        mean_AoA = np.mean(AoA,axis=0)
                        mean_phase_diff = np.mean(phase_diff, axis=0) 
                        mean_phase_diff = mean_phase_diff[1:]
                        ### check phase diff < pi ###
                        for i,p in enumerate(mean_phase_diff):
                            if p>np.pi:
                                mean_phase_diff[i] = p - 2*np.pi
                        
                        if self.v_rx==0:
                            eta = 0
                            f_d = mean_phase_diff[0]/(2*np.pi*self.T)
                            v = 0
                            x0 = [f_d, v, eta]
                        else:
                            eta, f_d, v = self.solve_system(mean_phase_diff,mean_AoA)
                            x0 = [f_d, v, eta]
                            x0 = self.check_initial_values(x0)

                        fd_err_rel = []
                        eta_err_rel = []
                        v_err_rel = []
                        fd_err_abs = []
                        eta_err_abs = []
                        v_err_abs = []
                        nls_t = []
                        # collect results for each number of available static paths (n_static)
                        for k in self.n_static_list:
                            start = time.time()
                            results = least_squares(self.system, x0, args=(mean_phase_diff[:k+2], mean_AoA[:k+2]))    
                            nls_t.append(time.time()-start)
                            if relative: 
                                fd_err_rel.append(abs((self.fd-np.mean(results.x[0]))/self.fd))   
                                if save_all:
                                    eta_err_rel.append(abs((self.eta-np.mean(results.x[2]))/self.eta))
                                    v_err_rel.append(abs((self.v_rx-np.mean(results.x[1]))/self.v_rx))
                            else:
                                fd_err_abs.append(abs((self.fd-np.mean(results.x[0]))/self.fd))   
                                if save_all:
                                    eta_err_abs.append(abs(np.rad2deg(self.eta)-np.rad2deg(np.mean(results.x[2]))))
                                    v_err_abs.append(abs(self.v_rx-np.mean(results.x[1])))
                        nls_time.append(nls_t)

                        # collect results for each interval length
                        if relative:
                            int_fd_err_rel.append(fd_err_rel)
                            if save_all:
                                int_eta_err_rel.append(eta_err_rel)
                                int_v_err_rel.append(v_err_rel)
                        else:
                            int_fd_err_abs.append(fd_err_abs)
                            if save_all:
                                int_eta_err_abs.append(eta_err_abs)
                                int_v_err_abs.append(v_err_abs)

                    # collect results for each AoA std
                    if relative:
                        a_fd_err_rel.append(int_fd_err_rel)
                        if save_all:
                            a_eta_err_rel.append(int_eta_err_rel)
                            a_v_err_rel.append(int_v_err_rel)
                    else:
                        a_fd_err_abs.append(int_fd_err_abs)
                        if save_all:
                            a_eta_err_abs.append(int_eta_err_abs)
                            a_v_err_abs.append(int_v_err_abs)

                # collect results for each value of SNR 
                if relative:
                    fd_error_rel.append(a_fd_err_rel)
                    if save_all:
                        eta_error_rel.append(a_eta_err_rel)
                        v_error_rel.append(a_v_err_rel)
                else:
                    fd_error_abs.append(a_fd_err_abs)
                    if save_all:
                        eta_error_abs.append(a_eta_err_abs)
                        v_error_abs.append(a_v_err_abs)

            # collect the total final results
            if relative:
                tot_fd_error_rel.append(fd_error_rel)
                if save_all:
                    tot_eta_error_rel.append(eta_error_rel)
                    tot_v_error_rel.append(v_error_rel)
            else:
                
                tot_fd_error_abs.append(fd_error_abs)
                if save_all:
                    tot_eta_error_abs.append(eta_error_abs)
                    tot_v_error_abs.append(v_error_abs)

        if save:
            if relative:
                tot_fd_error_rel = np.stack(tot_fd_error_rel)
                if save_all:
                    tot_eta_error_rel = np.stack(tot_eta_error_rel)
                    tot_v_error_rel = np.stack(tot_v_error_rel)
            else:
                tot_fd_error_abs = np.stack(tot_fd_error_abs)
                if save_all: 
                    tot_eta_error_abs = np.stack(tot_eta_error_abs)
                    tot_v_error_abs = np.stack(tot_v_error_abs)
            if self.l==0.005:
                if relative:
                    np.save(path+'fd_rel_fc60.npy',tot_fd_error_rel)
                    if save_all:
                        np.save(path+'eta_rel_fc60.npy',tot_eta_error_rel)
                        np.save(path+'v_rel_fc60.npy',tot_v_error_rel)
                else:
                    np.save(path+'fd_abs_fc60.npy',tot_fd_error_rel)
                    if save_all:
                        np.save(path+'eta_abs_fc60.npy',tot_eta_error_abs)
                        np.save(path+'v_abs_fc60.npy',tot_v_error_abs)
            elif self.l==0.0107:
                if relative:
                    np.save(path+'fd_rel_fc28.npy',tot_fd_error_rel)
                    if save_all:
                        np.save(path+'eta_rel_fc28.npy',tot_eta_error_rel)
                        np.save(path+'v_rel_fc28.npy',tot_v_error_rel)
                else:
                    np.save(path+'fd_abs_fc28.npy',tot_fd_error_abs)
                    if save_all:
                        np.save(path+'eta_abs_fc28.npy',tot_eta_error_abs)
                        np.save(path+'v_abs_fc28.npy',tot_v_error_abs)
            elif self.l==0.06:
                if relative:
                    np.save(path+'fd_rel_fc5.npy',tot_fd_error_rel)
                    if save_all:
                        np.save(path+'eta_rel_fc5.npy',tot_eta_error_rel)
                        np.save(path+'v_rel_fc5.npy',tot_v_error_rel)
                else:
                    np.save(path+'fd_abs_fc5.npy',tot_fd_error_abs)
                    if save_all:
                        np.save(path+'eta_abs_fc5.npy',tot_eta_error_abs)
                        np.save(path+'v_abs_fc5.npy',tot_v_error_abs)
            f = open(path + 'info.txt', 'w')
            f.write('numbers of static paths: ' + str(self.n_static_list) + '\n')
            f.write('interval lengths: ' + str(self.interval_list) + '\n')
            f.write('AoA std: ' + str(aoa) + '\n')
            f.write('SNR values: ' + str(self.snr_list) + '\n')
            f.write('number of iterations: ' + str(N))
            f.close()

        return tot_fd_error_rel
                  
if __name__=='__main__':
    ### parameters ###
    fc = 5 # 5 or 28 or 60 GHz
    l = 3e8/(fc*1e9) # check for 28
    n_static = [2]
    snr=[5,10,20]
    interval=[16]
    aoa=[5]
    ##################
    d_est = DopplerEst(l, n_static, snr, interval)
    err = d_est.simulation(path='cir_estimation_sim/data/test/', N=1000, aoa=aoa, save_all=True)
    print(np.mean(err, axis=0))
    utils.print_error('cir_estimation_sim/data/test/',fc)
    utils.plot('cir_estimation_sim/data/test/', fc, snr=snr, aoa=aoa, interval=interval, n_static=n_static, save=True)