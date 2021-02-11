import numpy as np
from scipy import signal, integrate
import scipy.constants as const


class Atom:
    def __init__(self,atomsym):
        self.atomSymbol = atomsym
    
    def assignVels(self,vels):
        self.velocities = vels
    
    def assignVcorr(self,vcorr):
        self.vcorr = vcorr
    
    def assignPowSpec(self,powspec):
        self.powspec = powspec
    
    def assignTimeList(self,timelist):
        self.times = timelist
    
    def assignFreqList(self,freqList):
        self.freqs = freqList
    
    def assignXS(self,XS):
        self.XS = XS
    
    def assignAtomNum(self,atomnum):
        self.atomNum = atomnum

    def assignTemperature(self, temp):
        self.T = temp

    def writePowSpecFile(self):
        filename = 'temp' + self.atomNum + 'powspec.txt'
        file_writer = open(filename,'w')
        for k in range(len(self.freqs)):
            file_writer.write(self.freqs[k])
            file_writer.write('\t')
            for i in range(3):
                for j in range(3):
                    file_writer.write(self.powspec[i,j,k])
                    file_writer.write('\t')
            file_writer.write('\n')
    
    def computeDWF(self,temp,isotropic_flag):
        self.freqs = np.where(self.freqs ==0, 1e-34, self.freqs)
        if isotropic_flag:
            u = np.zeros(len(self.freqs))
            for i in range(3):
                u = u + np.multiply(np.true_divide(2*10**12*const.hbar/(6*const.Boltzmann*temp),self.freqs),np.absolute(self.powspec[i,i,:]))
            A = integrate.simps(u[1:],self.freqs[1:])/(2*const.pi)
            # application of Q-dependence to DWF
            q_sq = self.computeQ_sq(1)
            DWF = np.exp(-q_sq*A)
        else:
            A = np.zeros([3,3])
            B = np.zeros([3,3,len(self.freqs)])
            for i in range(3):
                for j in range(3):
                    B[i,j,1:]  = np.multiply(np.true_divide(2*10**12*const.hbar/(2*const.Boltzmann*temp),self.freqs[1:]),self.powspec[i,j,1:])
                    A[i,j] = integrate.simps(B[i,j,1:],self.freqs[1:])/(2*const.pi)
            alpha = 0.2*(np.trace(A)+2*np.true_divide(np.absolute(np.trace(np.dot(A,B))),np.absolute(np.trace(B))))
            # application of Q-dependence to DWF
            q_sq = self.computeQ_sq(1)
            DWF = np.exp(-1.0*np.multiply(q_sq,alpha))
            DWF[0]=0.0
        return DWF

    def convolute(self,fun1,fun2):
        return signal.fftconvolve(fun1,fun2,'full')[:len(fun1)]

    def initSlaw(self):
        try:
            self.powspec
        except NameError:
            print("Error powspec not computed yet!\n")
        else:
            self.Slaw = np.zeros(len(self.powspec[0,0,:]))

    def createF1(self):
        self.f1 = 2*10**12*const.hbar/(2*self.freqs*const.Boltzmann*self.T)*self.powspec
        self.f1[:,:,0]=0.0
    
    def initializeFn(self):
        self.fn = self.f1
    
    def increaseFn(self):
        temp_fn = np.zeros([3,3,len(self.fn[0,0,:])],dtype=np.complex_)
        for i in range(3):
            for j in range(3):
                for ind in range(3):
                    temp_fn[i,j,:] = temp_fn[i,j,:] + self.freqs[1]*self.convolute(self.fn[i,ind,:],self.f1[ind,j,:])/(2*const.pi)
        self.fn = temp_fn
    
    def computeQ_sq(self, order):
        kf = np.sqrt(4*const.pi*const.m_n*32*(const.c*100)/const.hbar)*10**(-9)        # kf in 1/nm units
        alpha = np.sqrt(2*const.m_n*self.freqs*10**-6/(const.hbar) + kf**2)        # alpha defined as (2*m*(deltaE)/hbar^2 + kf^2)^.5 in 1/nm units
        theta = 3.0*const.pi/4.0                                                # only implement back scatter for now
        q_sq = np.square(alpha - kf*np.cos(theta)) + (kf*np.sin(theta))**2
        # currently implementing crude q-dependence -> severely screws up very low energy peaks
        return (q_sq)**order


