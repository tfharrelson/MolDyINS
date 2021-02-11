# -*- coding: utf-8 -*-
#"""
#Created on Wed Aug  8 12:29:36 2018
#@author: Tommy
#pseudocode
#
#inputs - number of atoms, atom types, neutron cross sections, number of time steps
#problems foreseen: need to map each atom index to atom type and cross section
#"""

import sys
import numpy as np
import os
import scipy.constants as const
import math
import subprocess
from scipy import signal
from scipy import integrate
import gc
from mpi4py import MPI

# code only used to taking gromacs output trajectory xvg files
ndx_file = str()
trr_file = str()
tpr_file = str()
elements_file = str()
atom_list = np.array([])
index_groups = np.array([])
n_timesteps = -1
T = 5 #initialize temperature to default value for INS
name = str()
ncore = 1                                                # default parallelized cores is 1 (serial version)
mem_core = 3e8                                                # default memory for 1 process is 500 Mb

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
    def writePowSpecFile(self):
        filename = 'temp'+self.atomNum+'powspec.txt'
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
            #A = np.trapz(u[1:],self.freqs[1:])/(2*const.pi)
            A = integrate.simps(u[1:],self.freqs[1:])/(2*const.pi)
            print("A in isotropic DWF:")
            print(A)
            print("u in isotropic DWF:")
            print(u)
            # application of Q-dependence to DWF
            q_sq = self.computeQ_sq(1)
            DWF = np.exp(-q_sq*A)
            #DWF = np.exp(-1.0*(self.freqs/(2*const.pi*5.01e-3))*A)
        else:
            A = np.zeros([3,3])
            B = np.zeros([3,3,len(self.freqs)])
            for i in range(3):
                for j in range(3):
                    #B[i,j,1:] = np.multiply(np.true_divide(2*10**12*const.hbar/(2*const.Boltzmann*temp),self.freqs[1:]),np.absolute(self.powspec[i,j,1:]))
                    #self.freqs[1:] = np.where(self.freqs[1:]==0, 1e-3, self.freqs[1:])
                    B[i,j,1:]  = np.multiply(np.true_divide(2*10**12*const.hbar/(2*const.Boltzmann*temp),self.freqs[1:]),self.powspec[i,j,1:])
                    #A[i,j] = np.trapz(B[i,j,1:],self.freqs[1:])/(2*const.pi)
                    A[i,j] = integrate.simps(B[i,j,1:],self.freqs[1:])/(2*const.pi)
            alpha = 0.2*(np.trace(A)+2*np.true_divide(np.absolute(np.trace(np.dot(A,B))),np.absolute(np.trace(B))))
            print('alpha = ')
            print(alpha)
            # application of Q-dependence to DWF
            q_sq = self.computeQ_sq(1)
            print('q^2 = ')
            print(q_sq)
            DWF = np.exp(-1.0*np.multiply(q_sq,alpha))
            DWF[0]=0.0
            #DWF = np.exp(-self.freqs*alpha/(2*const.pi*5.01e-3))
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
        self.f1 = 2*10**12*const.hbar/(2*self.freqs*const.Boltzmann*T)*self.powspec
        self.f1[:,:,0]=0.0
        #tr_fn_int = 0.0
        #for i in range(3):
        #    tr_fn_int=tr_fn_int+np.trapz(self.f1[i,i,:],self.freqs)
        #self.f1 = self.f1/tr_fn_int      # normalization of f1
    def initializeFn(self):
        #self.fn = np.zeros([3,3,len(self.powspec[0,0,:])])
        #self.fn[:,:,0]=1.0
        self.fn = self.f1
        #tr_fn_int = 0.0
        #for i in range(3):
        #    tr_fn_int=tr_fn_int+np.trapz(self.fn[i,i,:],self.freqs)
        #self.fn = self.fn/tr_fn_int
    def increaseFn(self):
        temp_fn = np.zeros([3,3,len(self.fn[0,0,:])],dtype=np.complex_)
        for i in range(3):
            for j in range(3):
                for ind in range(3):
                    temp_fn[i,j,:] = temp_fn[i,j,:] + self.freqs[1]*self.convolute(self.fn[i,ind,:],self.f1[ind,j,:])/(2*const.pi)
        self.fn = temp_fn
    def computeQ_sq(self,order):
        kf = np.sqrt(4*const.pi*const.m_n*32*(const.c*100)/const.hbar)*10**(-9)        # kf in 1/nm units
        alpha = np.sqrt(2*const.m_n*self.freqs*10**-6/(const.hbar) + kf**2)        # alpha defined as (2*m*(deltaE)/hbar^2 + kf^2)^.5 in 1/nm units
        theta = 3.0*const.pi/4.0                                                # only implement back scatter for now
        q_sq = np.square(alpha - kf*np.cos(theta)) + (kf*np.sin(theta))**2
        # currently implementing crude q-dependence -> severely screws up very low energy peaks
        #N = 1/(5.01e-3*2*const.pi)
        #return (N*self.freqs)**order
        return (q_sq)**order
            

def create_index_groups(val):
    global index_groups
    vals = np.array(val.split())
    index_groups = vals.astype(np.float)

def input_switcher(arg,val):
    if arg == 'trr':
        global trr_file
        trr_file = val
    elif arg == 'tpr':
        global tpr_file
        tpr_file = val
    elif arg == 'ndx':
        global ndx_file
        ndx_file = val
    elif arg == 'elements':
        global elements_file
        elements_file = val
    elif arg == 'atom_list':
        global atom_list
        atom_list = np.array(val.split())
    elif arg == 'index_groups':
        create_index_groups(val)
    elif arg == 'T':
        global T
        T = float(val)
    elif arg == 'name':
        global name
        name = val
    else:
        print('INVALID ARGUMENT')
    """
    switcher = {
            'trr': trr_file = val,
            'tpr': tpr_file = val,
            'ndx': ndx_file = val,
            'elements': elements_file = val,
            'atom_list': atom_list = np.array(val.split()),
            'index_groups': create_index_groups(val)
            }
    switcher.get(arg, 'Invalid argument')
    """

def readInputFile(inputfile):
    inputReader = open(inputfile,'r')
    for line in inputReader:
        words = line.split('=')
        tag = words[0].strip()
        value = words[1].strip()
        #for i in range(len(words[1:]))
        #    value = words[i+1].strip()
        input_switcher(tag,value)

def find_n_timesteps(traj_file):
    traj_reader = open(traj_file,'r')
    #count = 0
    timelist = np.array([])
    for line in traj_reader:
        if line[0]!='#' and line[0]!='@':
            #count = count+1
            timelist = np.append(timelist,float(line.split()[0]))
    return timelist

def parseVelocities(n_atoms):

#        this routine is for creating blocks of atoms to be read simultaneously by
#        rest of code.
#        Output is an array with size = [n_blocks, 2] where each row has both the
#                first and last column indices of the full vels.xvg file for a 
#                particular block.
    if n_timesteps < 0:
        print('num timesteps not computed yet!! compute first before running this routine')
    max_cols = int(np.floor(mem_core/(8.*n_timesteps)*size))
    max_atoms = int(np.floor(max_cols/3.))
    print("max_atoms = " + str(max_atoms))
    print("mpi size = " + str(size))
    #n_parblockcols = int(max_atoms*3/size)
    n_parblocks = int(np.ceil(float(n_atoms)/max_atoms))
    print("n_parblocks = " + str(n_parblocks))
    tot_cols = n_atoms*3
    blocks = np.zeros([size,n_parblocks,2],dtype=int)
    blocks[0,0,0]=1
    if max_cols > (tot_cols - blocks[0,0,0]):
        n_cols = int(np.floor(tot_cols/size))
    else:
        n_cols = int(np.ceil(max_atoms*3/size))
    blocks[0,0,1]=int(n_cols+blocks[0,0,0])
    i = 0
    for j in range(n_parblocks):
        for i in range(1,size):
            blocks[i,j,0]=blocks[i-1,j,1]
            if i == size-1 and j == n_parblocks-1:
                blocks[i,j,1] = int(tot_cols+1)
            else:
                blocks[i,j,1]=int(blocks[i,j,0]+n_cols)
        if max_cols > (tot_cols - blocks[-1,j,1]):
            n_cols = int(np.ceil(float(tot_cols-blocks[-1,j,1])/size))
        else:
            n_cols = int(np.ceil(max_atoms*3/size))
        if j+1 < n_parblocks:
            blocks[0,j+1,0]=blocks[-1,j,1]
            blocks[0,j+1,1]=blocks[0,j+1,0]+n_cols
    #blocks[n_blocks-1,0]=blocks[n_blocks-2,1]
    #blocks[n_blocks-1,1]=int(3*n_atoms+1)
    return blocks

# 10.08.18 Rest of routine is obsolete
#    vel_reader = open(velfilename,'r')
#    # initialize array of filenames to be returned
#    filenames = np.array([])
#    times = np.array([])
#    filewriters = np.array([])
#    # CAN ONLY INITIALIZE 1000 files at a time
#    initFlag = True
#    for i in range(n_atoms):                # initialize all file-writers
##        filewriters = np.append(filewriters,open('temp_'+str(i+1)+'.txt','w'))
#        filenames = np.append(filenames,'temp_'+str(i+1)+'.txt')
#    count = 0
#    for line in vel_reader:
#        if line[0]=='#' or line[0]=='@':
#            continue
#        # split line into velocity components for each atom
#        # REMEMBER: first element is time, NOT a vel component
#        vels = np.array(line.split())
#        times = np.append(times, float(vels[0]))
#        for i in range(n_atoms):
#            if initFlag == True:
#                filewriter = open('temp_'+str(i+1)+'.txt','w')
#            else:
#                filewriter = open('temp_'+str(i+1)+'.txt','a')
#            column = 3*i+1                # '+ 1' is accounting for time column
#            filewriter.write('{}\t{}\t{}\n'.format(vels[column],vels[column+1],vels[column+2]))
#        initFlag = False
#        count = count+1
#        print('percent done = ' + str(count/25000.0*100))
#    return (times, filenames)

def readVelocities(traj_filename,n_steps,init_index,fin_index):
    traj_reader = open(traj_filename,'r')
    n_cols = fin_index - init_index
    velocities = np.zeros([n_steps, n_cols])
    i=0
    #column = 3*init_index+1
    column = init_index
    for line in traj_reader:
        if line[0]=='#' or line[0]=='@':
            continue
        #print("length of line = ")
        #print(len(line))
        allnums = np.array(line.split())
        #print("allnums = ")
        #print(allnums)
        #print("column = ")
        #print(column)
        #print("ncols = ")
        #print(n_cols)
        nums = allnums[column:(column+n_cols)]
        nums = nums.astype(np.float)
        #print("nums = ")
        #print(nums)
        #velocities[i,0]=nums[0]
        velocities[i,:]=nums
        i=i+1
    return velocities

def readIndices(ndxFile,group):
    count = 0
    ndxReader = open(ndxFile,'r')
    indexList = np.array([])
    line = next(ndxReader)
    while count <= group:
        try:
            line = next(ndxReader)
        except StopIteration:
            break
        if line[0] == '[':
            count = count+1
            continue
        if count == group:
            nums = np.array(line.split())
            nums = nums.astype(np.float)
            indexList = np.append(indexList, nums)
    return indexList

def makeNdx(index):
    ndx_writer = open('temp.ndx','w')
    ndx_writer.write('[ single atom ]\n')
    ndx_writer.write(str(index))
    ndx_writer.write('\n')

def createVelocities(trrFile, tprFile, ndxFile,index,mpiflag):
    if mpiflag:
        fcncall = 'gmx_mpi traj -f '+trrFile+' -s '+tprFile+' -n '+ndxFile+' -ov '+'vels'+'.xvg <<< '+str(int(index))
    else:
        fcncall = 'gmx traj -f '+trrFile+' -s '+tprFile+' -n '+ndxFile+' -ov '+'vels'+'.xvg <<< '+str(int(index))
    print(fcncall)
    #subprocess.Popen(fcncall.split(),stdout=subprocess.PIPE)
    P=subprocess.Popen(fcncall, shell=True, executable='/bin/bash')
    output,error = P.communicate()
    print(error)
    #os.system(fcncall)
    return 'vels.xvg'

def findNeutronXS(atom,elements_file):
    elReader = open(elements_file,'r')
    for line in elReader:
        words = line.split()
        if words[1]==atom:
            return float(words[4])

def outerVcorrelationFunction(velocities, ntimesteps, totaln):
    vcorr = np.zeros([3,3,ntimesteps])
    outerproduct = np.zeros([3,3])
    for deltaT_index in range(ntimesteps):
        vcorr_t = np.zeros([3,3])
        for i in range(totaln-deltaT_index):
            for index1 in range(3):
                for index2 in range(index1,3):
                    outerproduct[index1,index2]=velocities[i,index1]*velocities[i+deltaT_index,index2]
                    outerproduct[index2,index1]=outerproduct[index1,index2]
            #vcorr_t = vcorr_t + np.dot(np.transpose(velocities[i,:]),velocities[i+deltaT_index,:])
            vcorr_t = vcorr_t + outerproduct
        vcorr[:,:,deltaT_index]=vcorr_t
    return vcorr

def findPowSpec(vcorr,nt):
    powspec = np.zeros([3,3,int(nt)])
    for i in range(3):
        for j in range(3):
            powspec[i,j,:] = np.abs(np.fft.rfft(vcorr[i,j,:]))
    return powspec

def findPowSpec2(velocities,delta_t, tf):
    num_vels = velocities.shape[0]
    num_freqs = int(np.floor(num_vels/2)+1)
    powspec = np.zeros([3,3,num_freqs])
    fft_velocities = np.zeros([3, num_freqs],dtype=np.complex_)
    N = 1.0
    #hanning_window = np.hanning(len(velocities[:,0]))*2/3*np.sqrt(6)                # factor needed to maintain the correct average magnitude of velocities
    for i in range(3):
        #using hanning window (comment out if window is not needed)
        #fft_velocities[i,:]=np.fft.rfft(np.multiply(velocities[:,i],hanning_window))
        fft_velocities[i,:]=np.fft.rfft(velocities[:,i])
    for i in range(3):
        for j in range(3):
            #powspec[i,j,:]=(delta_t**2/tf)*np.absolute(fft_velocities[i,:]*fft_velocities[j,:])
            powspec[i,j,:]=(delta_t**2/tf)*(np.conj(fft_velocities[i,:])*fft_velocities[j,:])
            #powspec[j,i,:]=powspec[i,j,:]
    return N*powspec

def getPowderAvgPrefactor(n):
    #return 1.0/((2.*n+1.)*math.factorial(n))
    #num = 0.0
    #prev_num = 1.0
    #prev_prev_num = 0.0
    #for i in range(2,2*int(n+1)+1):
    #    new_num = (i-1)/(i+1)*(2*num + 3*prev_num)
    #    prev_prev_num = prev_num
        #prev_num = num
    #    num = new_num
    #return prev_prev_num/(math.factorial(n)*num)
    #return (1.0/3.0)**n*(1.0/math.factorial(n))
    return (1.0/(3.0*math.factorial(n)))*(3.0/5.0)**(n-1)

def convoluteSpectrum(spec, freqs, deltaE):
    # convolute with Gaussian that increases variance with energy transfer
    # frequency units are cm-1 in this routine
    omegas = np.append(-1.0*freqs[::-1],freqs[1:])
    conv_spec = np.zeros([len(spec)])
    minres = 1.21        # minimum resolution of the VISION spectrometer in cm-1 units
    for i in range(len(freqs)):
        sigma = deltaE*freqs[i] + minres
        gaussian = 1/(sigma*np.sqrt(2*const.pi))*np.exp(-1.0/2*np.square(omegas/sigma))
        zero_list_back = np.zeros(len(freqs)-1-i)
        zero_list_front = np.zeros(i)
        spec_added_backzeros = np.append(zero_list_back,spec)
        spec_added_allzeros = np.append(spec_added_backzeros, zero_list_front)
        conv_spec[i]=np.dot(spec_added_allzeros,gaussian)
        #print('current sigma = ' + str(sigma))
        #print('dot product = ' + str(conv_spec[i]))
    return conv_spec
        
# start main code

input_file = sys.argv[1]
print("reading input file...")
readInputFile(input_file)
print("input file read!")
i=0
num_overtones = 10
mpi_Flag = 0
for atom in atom_list:
    # find cross section for the atom
    currAtom = Atom(atom)
    print("assigning XS...")
    currAtom.assignXS(findNeutronXS(atom,elements_file))
    print("XS assigned! Now reading indices...")
    curr_indices = readIndices(ndx_file,index_groups[i])
    print("indices read!")
    print("rank = ")
    print(rank)
    if rank == 0:
        print('Creating velocities using gmx traj')
        traj_file=createVelocities(trr_file,tpr_file,ndx_file,index_groups[i],mpi_Flag)
#        print("getting time list from velocity file...")
        #traj_file = 'vels.xvg'
        timelist = find_n_timesteps(traj_file)
        n_timesteps = len(timelist)
    else:
        n_timesteps = None
        traj_file = None
    print('broadcasting traj_file and n_timesteps...')
    traj_file=comm.bcast(traj_file,root=0)
    n_timesteps=comm.bcast(n_timesteps,root=0)
    print('success!')
    if rank != 0:
        print('displaying n timesteps and traj_file for rank ' + str(rank))
        print(n_timesteps)
        print(traj_file)
        timelist = np.empty(n_timesteps)
    comm.Bcast(timelist,root=0)
    n_blocks = None
    if rank == 0:
        print("getting blocks...")
        total_blocks = parseVelocities(len(curr_indices))
        print("blocks = ")
        print(total_blocks)
        n_blocks = int(len(total_blocks[0,:,0]))
        print(n_blocks)
        #n_blocks=5
    else:
        n_blocks = None
        total_blocks = None
    print(n_blocks)
    n_blocks=comm.bcast(n_blocks,root=0)
    blocks = np.empty([n_blocks,2],dtype=int)
    comm.Scatter(total_blocks,blocks,root=0)
    count = 0
    for block_index in range(len(blocks[:,0])):
        print("reading block velocities from vels.xvg")
        print("curr block = ")
        print(blocks[block_index,:])
        block_velocities = readVelocities(traj_file, n_timesteps,blocks[block_index,0],blocks[block_index,1])
        for index in range(int(len(block_velocities[0,:])/3)):
            currAtom.assignAtomNum(index)
            curr_col = 3*index
            velocities = block_velocities[:,curr_col:(curr_col+3)]
            currAtom.assignVels(velocities)
            currAtom.assignTimeList(timelist)
            currAtom.assignPowSpec(findPowSpec2(currAtom.velocities,currAtom.times[1],currAtom.times[-1]))
            try:
                currAtom.freqs
            except AttributeError:
                delta_w = 2.*const.pi/currAtom.times[-1]
                currAtom.assignFreqList(np.arange(len(currAtom.powspec[0,0,:]))*delta_w)
            print("done!")
            #currAtom.writePowSpecFile()
            # initialize scattering law for current atom
            kT_real = const.k*T
            kT_testx = np.trapz(2*currAtom.powspec[0,0,:],currAtom.freqs)*const.m_p/(2*const.pi)*10**(6)
            kT_testy = np.trapz(2*currAtom.powspec[1,1,:],currAtom.freqs)*const.m_p/(2*const.pi)*10**(6)
            kT_testz = np.trapz(2*currAtom.powspec[2,2,:],currAtom.freqs)*const.m_p/(2*const.pi)*10**(6)
            print("testing equipartition theorem and unit conversions...")
            print("kT = "+str(kT_real))
            print("kT_x = "+str(kT_testx))
            print("kT_y = "+str(kT_testy))
            print("kT_z = "+str(kT_testz))
            print("ratio x = "+str(kT_testx/kT_real))
            print("ratio y = "+str(kT_testy/kT_real))
            print("ratio z = "+str(kT_testz/kT_real))
            currAtom.initSlaw()
            #print("computing scattering law...")
            DWF_ai = currAtom.computeDWF(T,0)
            DWF_iso = currAtom.computeDWF(T,1)
            currAtom.createF1()
            currAtom.initializeFn()
            if i==0:
                overtones = np.zeros([len(currAtom.Slaw),num_overtones])
                if rank==0:
                    total_overtones = np.zeros([len(currAtom.Slaw),num_overtones])
            for n in range(1,num_overtones+1):
                print("on overtone number: " + str(n))
                #currAtom.increaseFn()
                pre = getPowderAvgPrefactor(n)
                q_sq = currAtom.computeQ_sq(n)
                if n==1:
                    currAtom.Slaw = currAtom.Slaw + np.absolute(pre*q_sq*np.trace(currAtom.fn)*DWF_ai)
                    overtones[:,n-1] = overtones[:,n-1] + np.absolute(pre*q_sq*np.trace(currAtom.fn)*DWF_ai)
                else:
                    currAtom.Slaw = currAtom.Slaw + np.absolute(pre*q_sq*np.trace(currAtom.fn)*DWF_iso)
                    overtones[:,n-1] = overtones[:,n-1] + np.absolute(pre*q_sq*np.trace(currAtom.fn)*DWF_iso)
                currAtom.increaseFn()
            if i == 0:
                total_S = np.zeros(len(currAtom.Slaw))
                if rank==0:
                    total_mpi_Slaw = np.zeros(len(currAtom.Slaw))
            total_S = total_S + currAtom.Slaw
            
            i=i+1
        block_velocities = []
        gc.collect()
    mpi_Slaw = None
    mpi_overtones = None
    if rank == 0:
        mpi_Slaw = np.empty([size,len(total_S)])
        mpi_overtones = np.empty([size,len(total_S),num_overtones])
        #mpi_Slaw=comm.Gather(total_S,mpi_Slaw)
    comm.Gather(total_S,mpi_Slaw, root=0)
    comm.Gather(overtones,mpi_overtones,root=0)
    if rank == 0:
        print('mpi_Slaw = ')
        print(mpi_Slaw)
        total_mpi_Slaw = total_mpi_Slaw + np.sum(mpi_Slaw,axis=0)
        total_overtones = total_overtones + np.sum(mpi_overtones,axis=0)
    
# NOT DONE YET FINISH THE GATHER CODE
# NEED TO INITIALIZE TOTAL MPI S_LAW FOR FIRST ATOM
# NEED TO ADD SUM OF GATHERED SLAWS TO TOTAL
# write out the freq list and total_S in separate columns
# convert freqs to cm-1
if rank == 0:
    q_sq = currAtom.computeQ_sq(1)
    currAtom.freqs = currAtom.freqs*(100.0/(6*const.pi))
    data = np.zeros([len(total_S)-1,2+num_overtones])
    data[:,0] = currAtom.freqs[1:]
    data[:,1] = total_mpi_Slaw[1:]
    #data[:,1]=np.trace(currAtom.f1)[1:]
    #test = np.convolve(np.trace(currAtom.f1),np.trace(currAtom.f1),'full')
    #test = np.trace(currAtom.f1)
    #print('integral of powspec trace:')
    #print(np.trapz(np.trace(currAtom.powspec),currAtom.freqs))
    #print('integral of f1:')
    #print(np.trapz(test,currAtom.freqs))
    #print('average v^2')
    #print(np.mean(velocities**2))
    np.savetxt(name + '_MD_INS.txt',data)
    #np.savetxt('test_f2.txt',test)
    #DWF_data = np.zeros([len(DWF_iso),2])
    #DWF_data[:,0]=currAtom.freqs
    #DWF_data[:,1]=DWF_iso
    #np.savetxt('DWF.txt',DWF_data)
    data = np.zeros([len(total_S),2+num_overtones])
    data[:,0]=currAtom.freqs
    #gaussian = np.zeros(1001)
    #sigma = 5.0
    #omegas = np.append(-1.0*currAtom.freqs[::-1],currAtom.freqs[1:])
    #gaussian = 1/np.sqrt(2*(sigma**2)*const.pi)*np.exp(-1.*omegas**2/(2*sigma**2))
    #conv_S = np.convolve(gaussian,total_mpi_Slaw)
    #data[:,1]=conv_S[(len(total_mpi_Slaw)):(2*len(total_mpi_Slaw)-1)]
    # convolute all overtones
    deltaE = 0.01        # resolution approximation to VISION instrument (deltaE/E)=0.01
    data[:,1]=convoluteSpectrum(total_mpi_Slaw,currAtom.freqs,deltaE)
    print('convoluted spectrum = ')
    print(data[:,1])
    for i in range(num_overtones):
        data[:,i+2]=convoluteSpectrum(total_overtones[:,i],currAtom.freqs,deltaE)
        print('spectrum for overtone ' + str(i))
        print(data[:,i+2])
        #data[:,i+2]=signal.fftconvolve(gaussian,total_overtones[:,i])[(len(total_overtones[:,0])):(2*len(total_overtones[:,0])-1)]
    np.savetxt(name+'_convGauss.txt',data)
    #q_sq = currAtom.computeQ_sq(1)
    data2 = np.empty([len(currAtom.freqs),2])
    data2[:,0]=currAtom.freqs
    data2[:,1]=np.sqrt(q_sq)
    np.savetxt('qsq_test.txt',data2)
    data3 = np.empty([len(currAtom.powspec[0,0,:]),2])
    data3[:,0]=currAtom.freqs
    data3[:,1]=currAtom.powspec[0,0,:]
    np.savetxt('powspec_xx_test.txt',data3)
    # clear out temp files
    #os.system('rm vels.xvg')
