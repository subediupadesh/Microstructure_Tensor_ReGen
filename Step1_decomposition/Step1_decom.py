import numpy as np
import time
import os

path = os.path.abspath("")

dense_tensor = np.load('Mircorstucture_tensor.npy')


###############################
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

################################################################

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(int(tensor_size.shape[0])):
        if i != mode:
            index.append(int(i))
    size = []
    for i in index:
        size.append(int(tensor_size[i]))
    return np.moveaxis(np.reshape(mat, size, order = 'F'), 0, mode)

################################################################

def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

################################################################

def update_cg(var, r, q, Aq, rold):
    alpha = rold / np.inner(q, Aq)
    var = var + alpha * q
    r = r - alpha * Aq
    rnew = np.inner(r, r)
    q = r + (rnew / rold) * q
    return var, r, q, rnew

################################################################

def ell(ind_mode, f_mat, mat):
    return ((f_mat @ mat.T) * ind_mode) @ mat

################################################################

def conj_grad(sparse_tensor, ind, fact_mat, mode, maxiter = 5):
    dim, rank = fact_mat[mode].shape
    ind_mode = ten2mat(ind, mode)
    f = np.reshape(fact_mat[mode], -1, order = 'F')
    temp = []
    for k in range(3):
        if k != mode:
            temp.append(fact_mat[k])
    mat = kr_prod(temp[-1], temp[0])
    r = np.reshape(ten2mat(sparse_tensor, mode) @ mat
                   - ell(ind_mode, fact_mat[mode], mat), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (dim, rank), order = 'F')
        Aq = np.reshape(ell(ind_mode, Q, mat), -1, order = 'F')
        alpha = rold / np.inner(q, Aq)
        f, r, q, rold = update_cg(f, r, q, Aq, rold)
    return np.reshape(f, (dim, rank), order = 'F')

################################################################

def cp_decompose(dense_tensor, sparse_tensor, rank, maxiter = 50):
    dim = sparse_tensor.shape
    fact_mat, rmse = [], []
    for k in range(3):
        fact_mat.append(0.01 * np.random.randn(dim[k], rank))
    ind = sparse_tensor != 0
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    show_iter = 10
    for it in range(maxiter):
        for k in range(3):
            fact_mat[k] = conj_grad(sparse_tensor, ind, fact_mat, k)
        tensor_hat = np.einsum('ur, vr, xr -> uvx', 
                               fact_mat[0], fact_mat[1], fact_mat[2])
        rse = (np.linalg.norm(tensor_hat[pos_test] - dense_tensor[pos_test], 2) / np.linalg.norm(dense_tensor[pos_test], 2))
        rmse.append(rse)
        if (it + 1) % show_iter == 0:
            print(f'Iter: {it+1}')
            print(rse)
            print()
    return tensor_hat, fact_mat, rmse

#######################################

def random_tensor_generator(p, shape):
  np.random.seed(1)
  random_tensor = np.zeros(shape)
  num_ones_per_subarray = int(p * random_tensor.size // (random_tensor.shape[0] * random_tensor.shape[1]))
  for i in range(random_tensor.shape[0]):
      for j in range(random_tensor.shape[1]):
          positions = np.random.choice(random_tensor.shape[2], num_ones_per_subarray, replace=False)
          random_tensor[i, j, positions] = 1
        
  return random_tensor

#######################################
# np.random.seed(1)

# x_dim, y_dim, t_dim = dense_tensor.shape
# random_tensor = np.random.rand(x_dim, y_dim, t_dim)

######################################
####### Sparse with 10% ##############
######################################
# p = 0.9
# sparse_tensor_1 = dense_tensor * np.round(random_tensor + 0.5 - p)
p1 = 0.10
sparse_tensor_1 = dense_tensor * random_tensor_generator(p1, dense_tensor.shape)

start = time.time()
rank = 100
recon_tensor_1, fact_mat_1, rmse_1 = cp_decompose(dense_tensor, sparse_tensor_1, rank, maxiter = 500)
end = time.time()
print(f'Running time: {end - start} seconds')

np.save(path+'/reconstructed_tensors/recon_cpd_tensor_1.npy', recon_tensor_1)
np.save(path+'/reconstructed_tensors/rmse_cpd_1.npy', rmse_1)

##########################################

######################################
####### Sparse with 15% ##############
######################################
# p = 0.85
# sparse_tensor_2 = dense_tensor * np.round(random_tensor + 0.5 - p)
p2 = 0.15
sparse_tensor_2 = dense_tensor * random_tensor_generator(p2, dense_tensor.shape)

start = time.time()
rank = 100
recon_tensor_2, fact_mat_2, rmse_2 = cp_decompose(dense_tensor, sparse_tensor_2, rank, maxiter = 500)
end = time.time()
print(f'Running time: {end - start} seconds')

np.save(path+'/reconstructed_tensors/recon_cpd_tensor_2.npy', recon_tensor_2)
np.save(path+'/reconstructed_tensors/rmse_cpd_2.npy', rmse_2)

#########################################

######################################
####### Sparse with 20% ##############
######################################
# p = 0.80
# sparse_tensor_3 = dense_tensor * np.round(random_tensor + 0.5 - p)
p3 = 0.20
sparse_tensor_3 = dense_tensor * random_tensor_generator(p3, dense_tensor.shape)

start = time.time()
rank = 100
recon_tensor_3, fact_mat_3, rmse_3 = cp_decompose(dense_tensor, sparse_tensor_3, rank, maxiter = 500)
end = time.time()
print(f'Running time: {end - start} seconds')

np.save(path+'/reconstructed_tensors/recon_cpd_tensor_3.npy', recon_tensor_3)
np.save(path+'/reconstructed_tensors/rmse_cpd_3.npy', rmse_3)





##########################################
############### Tucker Decomposition #####

##########################################
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
##########################################
def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(int(tensor_size.shape[0])):
        if i != mode:
            index.append(int(i))
    size = []
    for i in index:
        size.append(int(tensor_size[i]))
    return np.moveaxis(np.reshape(mat, size, order = 'F'), 0, mode)
##########################################

def update_cg(var, r, q, Aq, rold):
    alpha = rold / np.inner(q, Aq)
    var = var + alpha * q
    r = r - alpha * Aq
    rnew = np.inner(r, r)
    q = r + (rnew / rold) * q
    return var, r, q, rnew
##########################################
def ell_u(ind1, G, U, x_kron_v, lambda0):
    return ((U @ G @ x_kron_v.T) * ind1) @ x_kron_v @ G.T + lambda0 * U
##########################################
def conj_grad_u(mat1, ind1, G, U, V, X, lambda0, maxiter = 5):
    dim1, dim2 = U.shape
    u = np.reshape(U, -1, order = 'F')
    x_kron_v = np.kron(X, V)
    r = np.reshape(mat1 @ x_kron_v @ G.T
                   - ell_u(ind1, G, U, x_kron_v, lambda0), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (dim1, dim2), order = 'F')
        Aq = np.reshape(ell_u(ind1, G, Q, x_kron_v, lambda0), -1, order = 'F')
        u, r, q, rold = update_cg(u, r, q, Aq, rold)
    return np.reshape(u, (dim1, dim2), order = 'F')
##########################################
def ell_v(ind2, G2, x_kron_u, V, lambda0):
    return ((V @ G2 @ x_kron_u.T) * ind2) @ x_kron_u @ G2.T + lambda0 * V
##########################################
def conj_grad_v(mat2, ind2, G2, U, V, X, lambda0, maxiter = 5):
    dim1, dim2 = V.shape
    v = np.reshape(V, -1, order = 'F')
    x_kron_u = np.kron(X, U)
    r = np.reshape(mat2 @ x_kron_u @ G2.T
                   - ell_v(ind2, G2, x_kron_u, V, lambda0), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (dim1, dim2), order = 'F')
        Aq = np.reshape(ell_v(ind2, G2, x_kron_u, Q, lambda0), -1, order = 'F')
        v, r, q, rold = update_cg(v, r, q, Aq, rold)
    return np.reshape(v, (dim1, dim2), order = 'F')
##########################################
def ell_x(ind3, G3, v_kron_u, X, lambda0):
    return ((X @ G3 @ v_kron_u.T) * ind3) @ v_kron_u @ G3.T + lambda0 * X
##########################################
def conj_grad_x(mat3, ind3, G3, U, V, X, lambda0, maxiter = 5):
    dim1, dim2 = X.shape
    x = np.reshape(X, -1, order = 'F')
    v_kron_u = np.kron(V, U)
    r = np.reshape(mat3 @ v_kron_u @ G3.T
                   - ell_x(ind3, G3, v_kron_u, X, lambda0), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (dim1, dim2), order = 'F')
        Aq = np.reshape(ell_x(ind3, G3, v_kron_u, Q, lambda0), -1, order = 'F')
        x, r, q, rold = update_cg(x, r, q, Aq, rold)
    return np.reshape(x, (dim1, dim2), order = 'F')
##########################################
def ell_g(ind1, G, U, x_kron_v, lambda0):
    return U.T @ ((U @ G @ x_kron_v.T) * ind1) @ x_kron_v + lambda0 * G
##########################################
def conj_grad_g(mat1, ind1, G, U, V, X, lambda0, maxiter = 5):
    dim1, dim2 = G.shape
    g = np.reshape(G, -1, order = 'F')
    x_kron_v = np.kron(X, V)
    r = np.reshape(U.T @ mat1 @ x_kron_v - ell_g(ind1, G, U, x_kron_v, lambda0), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (dim1, dim2), order = 'F')
        Aq = np.reshape(ell_g(ind1, Q, U, x_kron_v, lambda0), -1, order = 'F')
        g, r, q, rold = update_cg(g, r, q, Aq, rold)
    return np.reshape(g, (dim1, dim2), order = 'F')
##########################################

def unfold(tensor, ind, G, mode):
    return ten2mat(tensor, mode), ten2mat(ind, mode), ten2mat(G, mode)
##########################################
def tucker_decomposition(dense_tensor, sparse_tensor, rank, lambda0, maxiter):
    dim = sparse_tensor.shape
    G = np.random.randn(rank[0], rank[1], rank[2])
    U = np.random.randn(dim[0], rank[0])
    V = np.random.randn(dim[1], rank[1])
    X = np.random.randn(dim[2], rank[2])
    ind = sparse_tensor != 0
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    tensor_hat = sparse_tensor.copy()
    show_iter = 10
    rmse = []
    for it in range(maxiter):
        mat1, ind1, G1 = unfold(sparse_tensor, ind, G, 0)
        G1 = conj_grad_g(mat1, ind1, G1, U, V, X, lambda0)
        U = conj_grad_u(mat1, ind1, G1, U, V, X, lambda0)
        G = mat2ten(G1, rank, 0)
        mat2, ind2, G2 = unfold(sparse_tensor, ind, G, 1)
        V = conj_grad_v(mat2, ind2, G2, U, V, X, lambda0)
        mat3, ind3, G3 = unfold(sparse_tensor, ind, G, 2)
        X = conj_grad_x(mat3, ind3, G3, U, V, X, lambda0)
        temp = ten2mat(np.einsum('rtq, sq -> rts', 
                                 np.einsum('rpq, tp -> rtq', G, V), X), 0)
        tensor_hat = mat2ten(U @ temp, np.array(dim), 0)


        rse = (np.linalg.norm(tensor_hat[pos_test] - dense_tensor[pos_test], 2) / np.linalg.norm(dense_tensor[pos_test], 2))
        rmse.append(rse)
        if (it + 1) % show_iter == 0:
            print(f'Iter: {it+1}')
            print(rse)
            print()
    return tensor_hat, G, U, V, X, rmse
# ##########################################

##########################################
def random_tensor_generator(p, shape):
  np.random.seed(1)
  random_tensor = np.zeros(shape)
  num_ones_per_subarray = int(p * random_tensor.size // (random_tensor.shape[0] * random_tensor.shape[1]))
  for i in range(random_tensor.shape[0]):
      for j in range(random_tensor.shape[1]):
          positions = np.random.choice(random_tensor.shape[2], num_ones_per_subarray, replace=False)
          random_tensor[i, j, positions] = 1
        
  return random_tensor
##########################################

# np.random.seed(1)

# x_dim, y_dim, t_dim = dense_tensor.shape
# random_tensor = np.random.rand(x_dim, y_dim, t_dim)

##########################################
# p = 0.9
# sparse_tensor_1 = dense_tensor * np.round(random_tensor + 0.5 - p)
p1 = 0.10
sparse_tensor_1 = dense_tensor * random_tensor_generator(p1, dense_tensor.shape)

import time
start = time.time()
rank = np.array([100, 100, 100])
lambda0 = 1e-4
recon_tuc_tensor_1, G, U, V, X, rmse_1 = tucker_decomposition(dense_tensor, sparse_tensor_1, rank, lambda0, maxiter=500)
end = time.time()
print('Running time: %d seconds'%(end - start))
np.save(path+'/reconstructed_tensors/recon_tuc_tensor_1.npy', recon_tuc_tensor_1)
np.save(path+'/reconstructed_tensors/rmse_ttd_1.npy', rmse_1)
##########################################

##########################################
# p = 0.85
# sparse_tensor_2 = dense_tensor * np.round(random_tensor + 0.5 - p)
p2 = 0.15
sparse_tensor_2 = dense_tensor * random_tensor_generator(p2, dense_tensor.shape)

import time
start = time.time()
rank = np.array([100, 100, 100])
lambda0 = 1e-4
recon_tuc_tensor_2, G, U, V, X, rmse_2 = tucker_decomposition(dense_tensor, sparse_tensor_2, rank, lambda0, maxiter=500)
end = time.time()
print('Running time: %d seconds'%(end - start))
np.save(path+'/reconstructed_tensors/recon_tuc_tensor_2.npy', recon_tuc_tensor_2)
np.save(path+'/reconstructed_tensors/rmse_ttd_2.npy', rmse_2)
##########################################

##########################################
# p = 0.8
# sparse_tensor_3 = dense_tensor * np.round(random_tensor + 0.5 - p)

p3 = 0.20
sparse_tensor_3 = dense_tensor * random_tensor_generator(p3, dense_tensor.shape)

import time
start = time.time()
rank = np.array([100, 100, 100])
lambda0 = 1e-4
recon_tuc_tensor_3, G, U, V, X, rmse_3 = tucker_decomposition(dense_tensor, sparse_tensor_3, rank, lambda0, maxiter=500)
end = time.time()
print('Running time: %d seconds'%(end - start))
np.save(path+'/reconstructed_tensors/recon_tuc_tensor_3.npy', recon_tuc_tensor_3)
np.save(path+'/reconstructed_tensors/rmse_ttd_3.npy', rmse_3)
##########################################