# MolDyINS
Code for converting Gromacs molecular dynamics trajectories into inelastic neutron scattering spectra generated at the indirect geometry spectrometers like VISION at ORNL or TOSCA at ISIS. The code is set up for calculation of the dynamic structure factor within the incoherent approximation. Full details regarding our formulation and approximations is [here](https://doi.org/10.1038/s41598-021-86771-5). The code is not available on pip or conda at the moment. 

### Prerequisites
- Numpy
- Scipy
- MPI4py
- Gromacs installed with 'gmx' or 'gmx_mpi' in the PATH

### Installation

We recommend that you create your own virtual environment with the above prerequisites. We like to use conda (see below), but other virtual environments also work fine:

`conda create -n myenv numpy scipy mpi4py`

This command creates a conda virtual environment named `myenv` with the prerequisites; feel free to change the name as you see fit. Activate the environment using:

`conda activate myenv`

Then clone this repository onto the machine that you want to run the calculation using:

`git clone git@github.com:tfharrelson/MolDyINS`

### Usage

`python MolDyINS.py mdins_input.txt`

where `mdins_input.txt` is an input file containing user-defined options (see more below) that control the action of the code.

### Input File Format

The general format is a text file with lines in the following format:

`keyword = user-defined option`

The allowed keywords are:

- `name`: The name of the output file
- `trr`: The gromacs trajectory file in .trr format
- `tpr`: The topology/input file used in the gromacs simulation in .tpr format
- `ndx`: The index file used in the gromacs simulation
- `elements`: The location and name of the elements file that contains the list of scattering cross-sections for each element
- `atom_list`: The atom types to be included in the MD_INS calculation (e.g. atom_list = C H)
- `index_groups`: The group numbers in the index file that are to be included in the MD_INS calculation. The number of entries here must match the number of entries in the `atom_list` keyword.
- `T`: The temperature of the molecular dynamics simulation  

### Notes

Sometimes it is convenient to have two different index groups that are the same element. For example, it might make sense to group hydrogens by their chemically equivalent positions. Then, there would be multiple index groups containing solely hydrogen. This is allowed by the code, but the `atom_list` input may look a little funny. If there are two different hydrogen groups, then the minimal inputs would be:

- `atom_list = H H`
- `index_groups = (index group #1) (index group #2)`

The user-defined values specified in the `atom_list` keyword are used to determine the neutron cross-section in the `elements.txt` file, and the index groups are used to get the relevant velocity data from the trajectory file (trr).

There exists a way to "hack" the code to arbitrarily amplify, or suppress the contribution from specific atoms. To do so, one can add a line to the `elements.txt` file to specify an arbitrary cross-section for a fictitious atom, and then use that fictitious atom type in the `atom_list` input command. This may prove useful for to specify isotope effects in INS experiments, or for general analytical purposes (e.g. to see the impact of atoms that do not have large neutron cross-sections in a spectrum).

### Citing us

If you use this code, please cite our article in Scientific Reports located [here](https://doi.org/10.1038/s41598-021-86771-5).

If using BibTeX, then feel free to add the following text to the bibliography file.

> @article{harrelson_scirep21,
> 
> author = {Harrelson, Thomas F. and Dettmann, Makena and Scherer, Christoph and Andrienko, Denis and Moul{\'{e}}, Adam J. and Faller, Roland},
> 
> doi = {10.1038/s41598-021-86771-5},
> 
> issn = {2045-2322},
> 
> journal = {Scientific Reports},
> 
> month = {apr},
> 
> number = {1},
> 
> pages = {7938},
> 
> title = {{Computing inelastic neutron scattering spectra from molecular dynamics trajectories}},
> 
> url = {http://www.nature.com/articles/s41598-021-86771-5},
> 
> volume = {11},
> 
> year = {2021}
> 
> }
