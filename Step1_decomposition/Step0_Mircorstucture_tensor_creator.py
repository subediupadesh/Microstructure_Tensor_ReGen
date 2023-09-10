import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

############################################################
### Opening Exodus File (Output result from simulation)

path = os.path.abspath('../Step0_phase_field_simulation/exodus_files/Ti_Cr.e')

model = netCDF4.Dataset(path)

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

npa = df_c.to_numpy().flatten().reshape(x_dim, y_dim, t_dim)
npa = np.rot90(npa)

np.save('Mircorstucture_tensor.npy', npa)

############################################################
### Plotting the last frame of microstructure evolution 

plt.rcParams["figure.figsize"] = [8,4]
plt.rcParams["figure.autolayout"] = True
ax1 = plt.subplot(121)
ax1.imshow(npa[:,:,-1], cmap='gist_ncar')
plt.show()