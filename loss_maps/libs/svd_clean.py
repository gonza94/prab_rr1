import numpy as np

def svd_decomposition(bpm_data, num):
    bpm_mean = np.mean(bpm_data)
    matrix = bpm_data - bpm_mean
    u, s, vt = np.linalg.svd(matrix/np.sqrt(matrix.shape[1]),full_matrices=False)
    svt_mat =  np.dot(np.sqrt(matrix.shape[1])*np.diag(s[:num]), vt[:num,:])
    u_mat = u[:,:num]
    clean_bpm = u_mat.dot(svt_mat) + bpm_mean
    return clean_bpm


