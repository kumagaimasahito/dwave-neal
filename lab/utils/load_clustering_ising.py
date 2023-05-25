from sklearn.datasets import make_blobs
from scipy.spatial import distance
import numpy as np

class load_clustering_ising:
    def __init__(self, n_clusters, n_points, random_state=0):
        self.n_clusters = n_clusters
        self.n_points = n_points
        self.random_state = random_state
    
    def _get_artificial_data(self):
        self.data, self.labels = make_blobs(random_state=self.random_state,
                                        n_samples=self.n_points,
                                        n_features=2, 
                                        cluster_std=1.5,
                                        centers=self.n_clusters)

    def print_matrix(mat):
        for i in range(len(mat)):
            for j in range(len(mat)):
                print(mat[i,j], end="   ")
            print(end="\n")

    def print_vector(vec):
        for i in range(len(vec)):
            print(vec[i], end="   ")
        print(end="\n")

    # def solution2labels(solution, n_clusters):
    #     labels = [i%n_clusters for i, n in enumerate(solution) if n==1]
    #     return labels

    # def cost(dist_mat, label):
    #     leng = len(dist_mat)
    #     cost = sum(
    #         [
    #             dist_mat[i,j]
    #             for i in range(0,leng)
    #             for j in range(i+1,leng)
    #             if label[i] == label[j]
    #         ]
    #     )
    #     return cost

    # def min_max(x, axis=None):
    #     x_min = x.min(axis=axis, keepdims=True)
    #     x_max = x.max(axis=axis, keepdims=True)
    #     return (x - x_min) / (x_max - x_min)


    def get(self):
        self._get_artificial_data() # file out data and labels

        self.dm = distance.cdist(self.data, self.data, metric="euclidean") # file out dist
        dm_min = self.dm.min(axis=None, keepdims=True)
        dm_max = self.dm.max(axis=None, keepdims=True)
        self.dist_mat = (self.dm - dm_min) / (dm_max - dm_min)

        self.lagr = self.n_points - self.n_clusters

    # create Jij matrix
        self.iden_mat = np.identity(self.n_clusters)
        self.ob_J_mat = 1/8 * np.kron(self.dist_mat, self.iden_mat)

        self.lagr_mat = np.diag(np.full(self.n_points, self.lagr))
        self.nden_mat = np.ones((self.n_clusters, self.n_clusters)) - self.iden_mat
        self.co_J_mat = 1/4 * np.kron(self.lagr_mat, self.nden_mat)

        self.J_mat = self.ob_J_mat + self.co_J_mat # file out J_mat

    # create hi vector
        self.di_h_mat = np.sum(self.dist_mat, axis=0)
        self.un_k_vec = np.ones(self.n_clusters)
        self.ob_h_vec = 1/4 * np.kron(self.di_h_mat, self.un_k_vec)

        self.un_N_vec = np.ones(self.n_points)
        self.co_h_vec = 1/2 * self.lagr * np.kron(self.un_N_vec, self.un_k_vec)

        self.h_vec = self.ob_h_vec + self.co_h_vec # file out h_vec

        return self.J_mat, self.h_vec

    def calc_ref_en(self):
        self.sol = - np.ones(self.n_clusters*self.n_points)
        for i, l in enumerate(self.labels):
            self.sol[i*self.n_clusters+l] = 1
        self.ref_en = self.sol @ self.J_mat @ self.sol.T + self.h_vec @ self.sol.T
        return self.ref_en
    
    def calc_en(self, state):
        return state @ self.J_mat @ state + self.h_vec @ state
