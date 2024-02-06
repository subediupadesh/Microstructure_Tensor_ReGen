import os
import glob
import netCDF4
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

##########################################################################################
path_nonelastic = os.path.abspath('../Step0_phase_field_simulation/nonelastic/exodus_files/Ti_Cr_non_elastic.e')
path_elastic = os.path.abspath('../Step0_phase_field_simulation/elastic/Cr_80P/exodus_files/Ti_Cr_elastic.e')

############################### For Model without elastic Effect #########################
model = netCDF4.Dataset(path_nonelastic)

X_all = model.variables['coordx'][:]
Y_all = model.variables['coordy'][:]
c = model.variables['vals_nod_var1'][:]

points = np.vstack([X_all,Y_all,c]).T
column_names = ['X', 'Y'] + list(range(c.shape[0]))

df = pd.DataFrame(points, columns=column_names)
df = df.sort_values(by = ['X', 'Y'], ascending = [True, True], ignore_index=True)
df_c = df.iloc[:, 2:]

x_dim = int(df_c.shape[0]**0.5)
y_dim = int(df_c.shape[0]**0.5)
t_dim = int(df_c.shape[1])

npa_noelastic = df_c.to_numpy().flatten().reshape(x_dim, y_dim, t_dim)
npa_noelastic = np.rot90(npa_noelastic)

np.save('noelastic_tensor.npy', npa_noelastic)

#############################################################################################


############################### For Model with elastic Effect #########################

path_csv = os.path.abspath('../Step0_phase_field_simulation/elastic/Cr_80P/exodus_files/csv_files/')
csv_files = natsorted(glob.glob(path_csv+'/*.csv'))  # Importing csv files of simulation naturally sorted

df_t0 = pd.read_csv(csv_files[0])
df_t0 = df_t0[['Points:0', 'Points:1']].rename(columns={'Points:0': 'X', 'Points:1': 'Y'})

array = []
for i, file in enumerate(csv_files):
  df = pd.read_csv(file, index_col=None)
  df = pd.concat([df_t0, df], axis=1)
  df = df.sort_values(by =['Y', 'X'], ascending = [True, True])
  dist = df[['Points:0', 'Points:1', 'c']]
  arr = dist.to_numpy()
  array.append(arr)

array = np.array(array)
x_dim = int(array.shape[1]**0.5)
y_dim = int(array.shape[1]**0.5)
t_dim = int(array.shape[0])
x_y_coord = array[:, :, :2]

npa_elastic_csv = np.transpose(array[:,:,2], (1,0))
npa_elastic_csv = npa_elastic_csv.reshape(x_dim, y_dim, t_dim) 
npa_elastic_csv = npa_elastic_csv[::-1, :, :]

np.save('elastic_tensor.npy', npa_elastic_csv)
np.save('xy_deformed.npy', x_y_coord)

######## Plotting last Frame #############
plt.rcParams["figure.figsize"] = [8,4]
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.imshow(npa_noelastic[:,:,-1], vmin=0.0, vmax=1.0, cmap='gist_ncar')
ax2.scatter(array[-1,:,0], array[-1,:,1], c= array[-1,:,2], vmin=0.0, vmax=1.0, cmap='gist_ncar')
ax1.set_title('No Elastic Effect')
ax2.set_title('Elastic Effect')
plt.show()
#############################################################################################














