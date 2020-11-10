import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Utility:
    INPUT_FILE = './data.txt'

    @staticmethod
    def parse_input():
        return np.genfromtxt(Utility.INPUT_FILE, autostrip = True)

class PCA:
    def __init__(self, input_data):
        self.input_data = input_data

    def perform_standardization(self):
        return StandardScaler().fit_transform(self.input_data)

    def compute_covariance_matrix(self, standardized_input_data):
        return np.cov(standardized_input_data.T)

    def compute_eigen_values_and_eigen_vectors(self, covariance_matrix):
        return np.linalg.eig(covariance_matrix)

    def sort_eigen_pairs(self, eigen_values, eigen_vectors):
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        eigen_pairs.sort(key = lambda x: x[0], reverse = True)
        return np.asarray(eigen_pairs, dtype = object)

    def compute_new_basis(self, sorted_eigen_pairs, k = 2):
        matrix_w = []
        for i in range(k + 1):
            matrix_w.append(sorted_eigen_pairs[i][1].reshape(-1, 1))
        return np.hstack(matrix_w)

    def compute_transformed_input(self):
        standardized_input_data = self.perform_standardization()
        covariance_matrix = self.compute_covariance_matrix(standardized_input_data)
        eigen_values, eigen_vectors = self.compute_eigen_values_and_eigen_vectors(covariance_matrix)
        sorted_eigen_pairs = self.sort_eigen_pairs(eigen_values, eigen_vectors)
        new_basis = self.compute_new_basis(sorted_eigen_pairs)
        transformed_input = np.dot(standardized_input_data, new_basis)
        return transformed_input
    
    def plot_data(self):
        transformed_input = self.compute_transformed_input()
        plt.scatter(transformed_input[:, 0], transformed_input[:, 1] * -1, marker = 'o', alpha = 0.6)
        plt.xlabel('PC-1')
        plt.ylabel('PC-2')
        plt.title('Transformed Input')
        plt.show()

class EM:
    def __init__(self):
        pass

def main():
    input_data = Utility.parse_input()
    pca_solver = PCA(input_data)
    pca_solver.plot_data()
    

if __name__ == "__main__":
    main()