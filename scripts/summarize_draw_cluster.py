'''
Sandro Tacchella
January 2, 2018   : iniate
run this script after run_make_SFH.py in order to
combine all (numpy) files to one (hdf5) file.

'''

# import modules

import numpy as np
import argparse
import os
import h5py

import read_in_halo_cat

from astropy.cosmology import WMAP7 as cosmo


# define paths

path_main = os.environ['WDIR_halo_model']
path_SFH_cat = path_main + 'catalogs/SFH/'


# get number of bins

parser = argparse.ArgumentParser()
parser.add_argument("--number_of_bins", type=int, help="number of cores")
parser.add_argument("--filename_SFH", type=str, help="filename of SFH file")
parser.add_argument("--redshift", type=float, help="redshift")
parser.add_argument("--DM_kind", type=str, help="DM kind (CDM or WDM)")
args = parser.parse_args()

number_of_bins = args.number_of_bins


# iterate over all files

counter = 0
idx_good = []

for ii in range(number_of_bins):
    try:
        file_name = path_SFH_cat + '/' + args.filename_SFH[:-5] + '/' + args.filename_SFH[:-5] + '_' + str(int(float(ii))) + '.npy'
        file_name_Mz = path_SFH_cat + '/' + args.filename_SFH[:-5] + '/' + args.filename_SFH[:-5] + '_Mz_' + str(int(float(ii))) + '.npy'
        file_name_Z = path_SFH_cat + '/' + args.filename_SFH[:-5] + '/' + args.filename_SFH[:-5] + '_Z_' + str(int(float(ii))) + '.npy'
        if (counter == 0):
            SFH_table_SFR = np.load(file_name)
            SFH_table_Mz = np.load(file_name_Mz)
            SFH_table_Z = np.load(file_name_Z)
            idx_good.append(ii)
        else:
            SFH_table_SFR = np.vstack([SFH_table_SFR, np.load(file_name)])
            SFH_table_Mz = np.vstack([SFH_table_Mz, np.load(file_name_Mz)])
            SFH_table_Z = np.vstack([SFH_table_Z, np.load(file_name_Z)])
            idx_good.append(ii)
    except IOError:
            SFH_table_SFR = np.vstack([SFH_table_SFR, -99.0*np.ones(len(SFH_table_SFR[-1]))])
            SFH_table_Mz = np.vstack([SFH_table_Mz, -99.0*np.ones(len(SFH_table_Mz[-1]))])
            SFH_table_Z = np.vstack([SFH_table_Z, -99.0*np.ones(len(SFH_table_Z[-1]))])
    counter += 1
    print 'progress (%): ', round(100.0*counter/number_of_bins, 3)


time_list = np.load(path_SFH_cat + '/' + args.filename_SFH[:-5] + '/' + args.filename_SFH[:-5] + '_t_' + str(int(float(idx_good[-1]))) + '.npy')


# get dark matter accretion history

z_table_in, M_table_in, Mt_table_in, is_contam, positions = read_in_halo_cat.read_in_halo_cat(args.redshift, args.DM_kind, cosmo, return_positions=True)

t_snapshots = 10**3*cosmo.age(z_table_in).value  # in Myr


# save combined array as hdf5 file

try:
    os.remove(path_SFH_cat + args.filename_SFH)
except OSError:
    pass


f = h5py.File(path_SFH_cat + args.filename_SFH, 'w')

# add SFH
grp_SFH = f.create_group("SFH")
grp_SFH.create_dataset('SFH_time', data=time_list)
grp_SFH.create_dataset('SFH_SFR', data=SFH_table_SFR)
grp_SFH.create_dataset('SFH_Mz', data=SFH_table_Mz)
grp_SFH.create_dataset('SFH_Z', data=SFH_table_Z)

# add DM assembly
grp_DM = f.create_group("DM")
grp_DM.create_dataset('DM_time', data=t_snapshots)
grp_DM.create_dataset('DM_z', data=z_table_in)
grp_DM.create_dataset('DM_cont', data=is_contam)
grp_DM.create_dataset('DM_M', data=M_table_in)
grp_DM.create_dataset('DM_Mt', data=Mt_table_in)
grp_DM.create_dataset('DM_pos', data=positions)
f.close()



