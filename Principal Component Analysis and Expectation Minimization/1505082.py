import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_normal

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
        return np.cov(standardized_input_data.T, bias = True)

    def compute_eigen_values_and_eigen_vectors(self, covariance_matrix):
        return np.linalg.eig(covariance_matrix)

    def sort_eigen_pairs(self, eigen_values, eigen_vectors):
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        eigen_pairs.sort(key = lambda x: x[0], reverse = True)
        return np.asarray(eigen_pairs, dtype = object)

    def compute_new_basis(self, sorted_eigen_pairs, k = 2):
        matrix_w = []
        for i in range(k):
            matrix_w.append(sorted_eigen_pairs[i][1].reshape(-1, 1))
        return np.hstack(matrix_w)

    def compute_transformed_input(self):
        covariance_matrix = self.compute_covariance_matrix(self.input_data)
        eigen_values, eigen_vectors = self.compute_eigen_values_and_eigen_vectors(covariance_matrix)
        sorted_eigen_pairs = self.sort_eigen_pairs(eigen_values, eigen_vectors)
        new_basis = self.compute_new_basis(sorted_eigen_pairs)
        transformed_input = np.dot(self.input_data, new_basis)
        return transformed_input
    
    def plot_data(self, transformed_input):
        plt.scatter(transformed_input[:, 0] , transformed_input[:, 1] * -1, marker = 'o', alpha = 0.6)
        plt.xlabel('PC-1')
        plt.ylabel('PC-2')
        plt.title('Transformed Input')
        plt.savefig('PCA.png')
        plt.show()

class EM:
    def __init__(self, data):
        self.data = data
        self.number_of_instances, self.D = data.shape
        self.K = 3  # 3 clusters from pca plot

    def compute_covariance_matrices(self):
        covariance_matrices = []
        for i in range(self.K):
            covariance_matrices.append(make_spd_matrix(n_dim = self.D))
        return np.asarray(covariance_matrices)

    def initialize_parmeters(self):
        random_instances = np.random.randint(low = 0, high = self.number_of_instances, size = self.K)
        self.means = np.asarray([self.data[instance, :] for instance in random_instances])
        self.covariance_matrices = self.compute_covariance_matrices()
        self.weights = np.full(shape = self.K, fill_value = 1 / self.K)

    def compute_normal_distribution(self, k):
        normal_distribution = multivariate_normal(self.means[k], self.covariance_matrices[k])
        return normal_distribution.pdf(self.data)

    def compute_log_likelihood(self):
        log_likelihood = 0.0
        for i in range(self.number_of_instances):
            inner_term = 0.0
            for k in range(self.K):
                N_k = multivariate_normal.pdf(x = self.data[i], mean = self.means[k], cov = self.covariance_matrices[k]);
                inner_term += self.weights[k] * N_k
            log_likelihood += np.log(inner_term)
        return log_likelihood

    def compute_conditional_distribution_of_latent_factors(self):
        probabilities = []
        for k in range(self.K):
            numerator = self.weights[k] * self.compute_normal_distribution(k).reshape(-1, 1)
            probabilities.append(numerator)
        probabilities = np.hstack(probabilities)
        probabilities /= np.sum(probabilities, axis = 1).reshape(-1, 1)
        return probabilities

    def e_step(self):
        probabilities = self.compute_conditional_distribution_of_latent_factors()
        return probabilities

    def update_mean(self, probabilities):
        numerator = np.matmul(probabilities.T, self.data)  # dimension = (K, D)
        denominator = np.sum(probabilities, axis = 0).reshape(-1, 1)  # dimension = (K, 1)
        self.means = numerator / denominator

    def update_covariance_matrices(self, probabilities):
        for k in range(self.K):
            numerator = np.zeros((self.D, self.D))
            for i in range(self.number_of_instances):
                difference = (self.data[i] - self.means[k]).reshape(-1, 1).T
                mul = np.matmul(difference.T, difference)
                numerator += probabilities[i][k] * np.matmul(difference.T, difference)
            denominator = np.sum(probabilities[:, k])
            self.covariance_matrices[k] = numerator / denominator

    def update_weights(self, probabilities):
        self.weights = np.sum(probabilities, axis = 0) / self.number_of_instances

    def m_step(self, probabilities):
        self.update_mean(probabilities)
        self.update_covariance_matrices(probabilities)
        self.update_weights(probabilities)

    @staticmethod
    def converged(current_log_likelihood, previous_log_likehood):
        EPS = 1e-6
        return abs(current_log_likelihood - previous_log_likehood) < EPS

    def perform_iterations(self, MAX_ITERATIONS = 100):
        self.initialize_parmeters()
        previous_log_likelihood, current_log_likelihood = -1.0, self.compute_log_likelihood()
        print("Initial Log Likelihood value =", current_log_likelihood)
        for i in range(MAX_ITERATIONS):
            probabilities = self.e_step()
            self.m_step(probabilities)
            previous_log_likelihood = current_log_likelihood
            current_log_likelihood = self.compute_log_likelihood()
            if(EM.converged(current_log_likelihood, previous_log_likelihood)):
                break
            print("Iteration =", i, " Log Likelihood value =", current_log_likelihood)
        
def main():
    input_data = Utility.parse_input()
    pca_solver = PCA(input_data)
    transformed_input = pca_solver.compute_transformed_input()
    pca_solver.plot_data(transformed_input)
    em_solver = EM(transformed_input)
    em_solver.perform_iterations()

if __name__ == "__main__":
    main()