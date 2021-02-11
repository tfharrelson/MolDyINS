import numpy as np
import subprocess
import math
import scipy.constants as const

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

def find_n_timesteps(traj_file):
    traj_reader = open(traj_file,'r')
    timelist = np.array([])
    for line in traj_reader:
        if line[0]!='#' and line[0]!='@':
            timelist = np.append(timelist,float(line.split()[0]))
    return timelist

def parseVelocities(n_atoms, n_timesteps, mem_core=3e8, size=1):

#   Create blocks of atoms to be passed to MPI ranks for parallel calculation of rest of code.
#
#   Params:
#   n_atoms:        number of atoms in the MD trajectory
#   n_timesteps:    number of time steps saved during the the MD simulation
#   mem_core:       max size of memory block used in each MPI process. Due to overhead concerns, choose a value that is at most half the actual max size. Default is 300 MB.
#   size:           number of MPI ranks used in calculation. Default is 1 (no MPI)
#
#   Return: an array with size = [n_blocks, 2] where each row has both the
#           first and last column indices of the full vels.xvg file for a 
#           particular block.
    if n_timesteps < 0:
        print('num timesteps not computed yet!! compute first before running this routine')
    max_cols = int(np.floor(mem_core/(8.*n_timesteps)*size))
    max_atoms = int(np.floor(max_cols/3.))
    print("max_atoms = " + str(max_atoms))
    print("mpi size = " + str(size))
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
    return blocks

def readVelocities(traj_filename,n_steps,init_index,fin_index):
    traj_reader = open(traj_filename,'r')
    n_cols = fin_index - init_index
    velocities = np.zeros([n_steps, n_cols])
    i=0
    column = init_index
    for line in traj_reader:
        if line[0]=='#' or line[0]=='@':
            continue
        allnums = np.array(line.split())
        nums = allnums[column:(column+n_cols)]
        nums = nums.astype(np.float)
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
    P=subprocess.Popen(fcncall, shell=True, executable='/bin/bash')
    output,error = P.communicate()
    print(error)
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
            powspec[i,j,:]=(delta_t**2/tf)*(np.conj(fft_velocities[i,:])*fft_velocities[j,:])
    return N*powspec

def getPowderAvgPrefactor(n):
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
    return conv_spec


