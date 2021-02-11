import sys
import numpy as np
import os
import scipy.constants as const
import gc
from mpi4py import MPI
from src.atom import Atom
from src.utils import *

class MolDyINS:
    def __init__(self, input_file):
        # Initialize fields for clarity
        self.trr_file = None
        self.tpr_file = None
        self.ndx_file = None
        self.elements_file = None
        self.atom_list = None
        self.index_groups = None
        self.T = None
        self.num_overtones = 10
        self.name = None
        self.ncore = 1
        self.mem_core = 3e8
        self.gmx_command = 'gmx'

        # Read inputs from file
        self.read_input_file(input_file)
    
    def read_input_file(self, inputfile):
        inputReader = open(inputfile,'r')
        for line in inputReader:
            words = line.split('=')
            tag = words[0].strip()
            value = words[1].strip()
            self.input_switcher(tag, value)
 
    def input_switcher(self, arg, val):
        if arg == 'trr':
            self.trr_file = val
        elif arg == 'tpr':
            self.tpr_file = val
        elif arg == 'ndx':
            self.ndx_file = val
        elif arg == 'elements':
            self.elements_file = val
        elif arg == 'atom_list':
            self.atom_list = np.array(val.split())
        elif arg == 'index_groups':
            self.create_index_groups(val)
        elif arg == 'T':
            self.T = float(val)
        elif arg == 'name':
            self.name = val
        elif arg == 'num_overtones':
            self.num_overtones = int(val)
        elif arg == 'ncore':
            self.ncore = int(val)
        elif arg == 'mem_core':
            self.mem_core = float(val)
        elif arg == 'gmx_command':
            self.gmx_command = val
        else:
            print('INVALID ARGUMENT')

    def create_index_groups(self, val):
        vals = np.array(val.split())
        self.index_groups = vals.astype(np.float)


# Initialize MPI4py objects
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# start main code

input_file = sys.argv[1]
print("reading input file...")
mdins = MolDyINS(input_file)
print("input file read!")
i=0
mpi_Flag = 0
for atom in mdins.atom_list:
    # find cross section for the atom
    currAtom = Atom(atom)
    currAtom.assignTemperature(mdins.T)
    print("assigning XS...")
    currAtom.assignXS(findNeutronXS(atom, mdins.elements_file))
    print("XS assigned! Now reading indices...")
    curr_indices = readIndices(mdins.ndx_file, mdins.index_groups[i])
    print("indices read!")
    print("rank = ")
    print(rank)
    if rank == 0:
        traj_file = createVelocities(mdins.trr_file, 
                                     mdins.tpr_file, 
                                     mdins.ndx_file, 
                                     mdins.index_groups[i], 
                                     mpi_Flag
                                    )
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
        total_blocks = parseVelocities(n_atoms=len(curr_indices), 
                                       n_timesteps=n_timesteps, 
                                       mem_core=mdins.mem_core, 
                                       size=size)
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
            # initialize scattering law for current atom
            kT_real = const.k * mdins.T
            kT_testx = np.trapz(currAtom.powspec[0,0,:], currAtom.freqs) * const.m_p / (2 * const.pi) * 10**(6)
            kT_testy = np.trapz(currAtom.powspec[1,1,:], currAtom.freqs) * const.m_p / (2 * const.pi) * 10**(6)
            kT_testz = np.trapz(currAtom.powspec[2,2,:], currAtom.freqs) * const.m_p / (2 * const.pi) * 10**(6)
            print("testing equipartition theorem and unit conversions...")
            print("kT = " + str(kT_real))
            print("kT_x = " + str(kT_testx))
            print("kT_y = " + str(kT_testy))
            print("kT_z = " + str(kT_testz))
            print("ratio x = " + str(kT_testx/kT_real))
            print("ratio y = " + str(kT_testy/kT_real))
            print("ratio z = " + str(kT_testz/kT_real))
            currAtom.initSlaw()
            DWF_ai = currAtom.computeDWF(mdins.T,0)
            DWF_iso = currAtom.computeDWF(mdins.T,1)
            currAtom.createF1()
            currAtom.initializeFn()
            if i==0:
                overtones = np.zeros([len(currAtom.Slaw), mdins.num_overtones])
                if rank==0:
                    total_overtones = np.zeros([len(currAtom.Slaw), mdins.num_overtones])
            for n in range(1, mdins.num_overtones+1):
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
        mpi_overtones = np.empty([size,len(total_S), mdins.num_overtones])
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
    currAtom.freqs = currAtom.freqs*(100.0 / (6 * const.pi))
    data = np.zeros([len(total_S)-1,2 + mdins.num_overtones])
    data[:,0] = currAtom.freqs[1:]
    data[:,1] = total_mpi_Slaw[1:]
    np.savetxt(mdins.name + '_MolDyINS.txt', data)
    data = np.zeros([len(total_S), 2 + mdins.num_overtones])
    data[:,0] = currAtom.freqs
    # convolute all overtones
    deltaE = 0.01        # resolution approximation to VISION instrument (deltaE/E)=0.01
    data[:,1]=convoluteSpectrum(total_mpi_Slaw,currAtom.freqs,deltaE)
    print('convoluted spectrum = ')
    print(data[:,1])
    for i in range(mdins.num_overtones):
        data[:,i+2]=convoluteSpectrum(total_overtones[:,i],currAtom.freqs,deltaE)
        print('spectrum for overtone ' + str(i))
        print(data[:,i+2])
        #data[:,i+2]=signal.fftconvolve(gaussian,total_overtones[:,i])[(len(total_overtones[:,0])):(2*len(total_overtones[:,0])-1)]
    np.savetxt(mdins.name+'_convGauss.txt',data)
    data2 = np.empty([len(currAtom.freqs),2])
    data2[:,0]=currAtom.freqs
    data2[:,1]=np.sqrt(q_sq)
    np.savetxt(mdins.name + 'qsq.txt',data2)
    data3 = np.empty([len(currAtom.powspec[0,0,:]), 2])
    data3[:,0]=currAtom.freqs
    data3[:,1]=currAtom.powspec[0,0,:]
    np.savetxt(mdins.name + 'powspec_xx.txt', data3)
