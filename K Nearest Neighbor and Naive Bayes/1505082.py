import os
import re
import ast
import sys
import html
import math
import time
import string
import pathlib
import traceback
import subprocess
from collections import Counter
from scipy.stats import ttest_rel
from xml.etree import ElementTree
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale

class Utility:
    Types = {"Train": 0, "Validation" : 1, "Test" : 2}
    document_set_sizes = {"Train" : 500, "Validation": 200, "Test": 500}

    folder_path = "./Data/Training"
    topics_file_path = "./Data/topics.txt"

    @staticmethod
    def install_module(module_name):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
        except subprocess.CalledProcessError:
            tb = traceback.format_exc()
        else:
            tb = "Successfully installed module: " + module_name
        finally:
            print(tb)

        Utility.global_import(module_name)


    @staticmethod
    def global_import(module_name):
        if "-" in module_name:
            module_name = module_name.replace("-", "_")

        globals()[module_name] = __import__(module_name)


    @staticmethod
    def import_modules():
        try:
            import pkg_resources
        except ImportError:
            Utility.install_module(pkg_resources)
            
        required = {"pandas", "numpy", "nltk", "bs4", "simplified-scrapy", "sklearn", "scipy"} 
        installed = {pkg.key for pkg in pkg_resources.working_set}
        for module_name in required:
            if module_name in installed:
                Utility.global_import(module_name)
            else:
                Utility.install_module(module_name)


    @staticmethod
    def download_nltk_resources():
        resources = ["punkt", "stopwords", "wordnet"]
        paths = ["tokenizers/punkt", "corpora/stopwords", "corpora/wordnet"]
        for resource, path in zip(resources, paths):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource)
            else:
                print(resource, "is already downloaded")


    @staticmethod
    def manage_dependencies():
        Utility.import_modules()
        Utility.download_nltk_resources()


    @staticmethod
    def convert_to_lower_case(text):
        return text.lower()


    @staticmethod
    def remove_digits(text):
        return text.translate({ord(k) : None for k in string.digits})


    @staticmethod
    def remove_punctuations(text):
        return text.translate(str.maketrans("", "", string.punctuation))


    @staticmethod
    def remove_html_tags(text):
        return bs4.BeautifulSoup(text, "html.parser").text

    
    @staticmethod
    def generate_word_tokens(line_text):
        return nltk.tokenize.word_tokenize(line_text)


    @staticmethod
    def remove_stopwords(words):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return [word for word in words if not word in stop_words]


    @staticmethod
    def perform_lemmatization(words):
        return [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]

    
    @staticmethod
    def perform_stemming(words):
        return [nltk.stem.PorterStemmer().stem(word) for word in words]


    @staticmethod
    def perform_preprocessing(line_text):
        line_text = Utility.remove_html_tags(line_text)
        line_text = Utility.convert_to_lower_case(line_text)
        line_text = Utility.remove_digits(line_text)
        line_text = Utility.remove_punctuations(line_text)
        words = Utility.generate_word_tokens(line_text)
        words = Utility.remove_stopwords(words)
        words = Utility.perform_lemmatization(words)
        words = Utility.perform_stemming(words)
        return words

    @staticmethod
    def is_non_empty_document(document):
        return numpy.asarray(document).any()

    @staticmethod
    def generate_topic_wise_documents(topic, all_unique_words):
        training_documents = []
        validation_documents = []
        testing_documents = []

        file_path = "./Data/Training/" + topic + ".xml"
        root = ElementTree.parse(file_path).getroot() 
        
        # training documents
        index = 0
        rows = root.findall('row')
        for row in rows[index : ]:
            document = Utility.perform_preprocessing(row.get('Body'))
            document = Utility.word_document_to_numeric_document(document, all_unique_words)
            if Utility.is_non_empty_document(document):
                training_documents.append(document)
                if len(training_documents) == Utility.document_set_sizes["Train"]:
                    break
            index += 1
        
        for row in rows[index : ]:
            document = Utility.perform_preprocessing(row.get('Body'))
            document = Utility.word_document_to_numeric_document(document, all_unique_words)
            if Utility.is_non_empty_document(document):
                validation_documents.append(document)
                if len(validation_documents) == Utility.document_set_sizes["Validation"]:
                    break
            index += 1

        for row in rows[index : ]:
            document = Utility.perform_preprocessing(row.get('Body'))
            document = Utility.word_document_to_numeric_document(document, all_unique_words)
            if Utility.is_non_empty_document(document):
                testing_documents.append(document)
                if len(testing_documents) == Utility.document_set_sizes["Test"]:
                    break
            index += 1

        return [training_documents, validation_documents, testing_documents]

    @staticmethod
    def does_file_exist(file_name, file_extension):
        file_path = "./" + file_name + file_extension
        return pathlib.Path(file_path).is_file()

    @staticmethod
    def load_data_from_file(file_name, file_extension):
        file_path = "./" + file_name + file_extension
        if file_extension == ".npy":
            data = numpy.load(file_path)
        elif file_extension == ".txt":
            with open(file_path, 'r', encoding = "utf-8") as f:
                data = f.read()
        return data

    @staticmethod
    def create_file(file_name, file_extension, data):
        file_path = "./" + file_name + file_extension
        if file_extension == ".npy":
            numpy.save(file_path, data)
        else:
            with open(file_path, 'w', encoding = "utf-8") as f:
                f.write(data)

    @staticmethod
    def generate_all_labels():
        label_matrices = []
        all_labels = [[], [], []] 

        with open(Utility.topics_file_path) as topics_file:
            for topic in topics_file:
                topic = topic.strip()
                if topic == "3d_Printer":
                    continue
                for (key, value), labels in zip(Utility.document_set_sizes.items(), all_labels):
                        labels += ([topic] * value)
        
        for labels in all_labels:
            label_matrix = numpy.hstack((labels))
            label_matrices.append(numpy.asarray(label_matrix))
        
        return label_matrices

    @staticmethod
    def generate_all_documents(all_unique_words):
        document_names = ["training_documents", "validation_documents", "testing_documents"]

        all_documents = [[], [], []]  # train, validation and test respectively
        label_matrices = Utility.generate_all_labels()
      
        document_matrices = []
          
        # check if numpy files are generated
        generated = True
        for document_name in document_names:
            if not Utility.does_file_exist(document_name, ".npy"):
               generated = False
               break

        if generated == True:
            for document_name in document_names:
                document_matrix = Utility.load_data_from_file(document_name, ".npy")
                document_matrices.append(document_matrix)
            return document_matrices, label_matrices
        
        with open(Utility.topics_file_path) as topics_file:
            for topic in topics_file:
                topic = topic.strip()
                if topic == "3d_Printer":
                    continue
        
                topic_wise_generated_documents = Utility.generate_topic_wise_documents(topic, all_unique_words)
                    
                for all_document, topic_wise_generated_document in zip(all_documents, topic_wise_generated_documents):
                    all_document.append(topic_wise_generated_document)

                    
        for all_document in all_documents:
            document_matrix = numpy.concatenate((all_document))
            document_matrices.append(document_matrix)
            
        for document_name, document_matrix in zip(document_names, document_matrices):
            Utility.create_file(document_name, ".npy", document_matrix)

        return document_matrices, label_matrices

    @staticmethod
    def word_document_to_numeric_document(document, all_unique_words):
        document_unique_words = set(document)
        frequency_count = Counter(document)
        return [frequency_count[word] if word in document_unique_words else 0 for word in all_unique_words]  # bag of words

    @staticmethod
    def generate_topic_wise_unique_words(topic):
        topic_wise_unique_words = set()
        
        file_path = "./Data/Training/" + topic + ".xml"
        root = ElementTree.parse(file_path).getroot() 

        total_size = Utility.document_set_sizes["Train"]

        # training documents
        count =  0
        rows = root.findall('row')
        for row in rows:
            word_list = Utility.perform_preprocessing(row.get('Body'))
            if word_list:
                topic_wise_unique_words |= set(word_list)
                count += 1
                if count == total_size:
                    break
        
        return topic_wise_unique_words

    @staticmethod
    def generate_all_unique_words():
        file_name = "all_unique_words"

        if(Utility.does_file_exist(file_name, ".txt")):
            all_unique_words = ast.literal_eval(Utility.load_data_from_file(file_name, ".txt"))
            return all_unique_words

        all_unique_words = set()
        with open(Utility.topics_file_path) as topics_file:
            for topic in topics_file:
                topic = topic.strip()
                if topic == "3d_Printer":
                    continue
                topic_wise_unique_words = Utility.generate_topic_wise_unique_words(topic)
                all_unique_words |= topic_wise_unique_words

        Utility.create_file(file_name, ".txt", str(all_unique_words))

        return all_unique_words

    @staticmethod
    def calculate_IDF(number_of_documents, count, alpha = 1e-6, beta = 1e-9):
        return numpy.log((alpha + number_of_documents) / (count + beta))

    @staticmethod
    def generate_data_for_cosine_similarity(train_matrix, test_matrix):
        TF_train = train_matrix.copy()
        TF_test = test_matrix.copy()

        TF_train = TF_train / numpy.sum(TF_train, axis = 1, keepdims = True)
        TF_test = TF_test / numpy.sum(TF_test, axis = 1, keepdims = True)

        IDF_train = numpy.sum((train_matrix.copy() > 0).astype('float'), axis = 0)
        IDF_train = numpy.asarray([Utility.calculate_IDF(len(train_matrix), count) for count in IDF_train])
        
        IDF_test = numpy.sum((test_matrix.copy() > 0).astype('float'), axis = 0)
        IDF_test = numpy.asarray([Utility.calculate_IDF(len(test_matrix), count) for count in IDF_test])

        return TF_train * IDF_train, TF_test * IDF_test   

    @staticmethod
    def get_topics():
        index_to_topics = defaultdict(str)
        index = 0
        with open(Utility.topics_file_path) as topics_file:
            for topic in topics_file:
                topic = topic.strip()
                if topic == "3d_Printer":
                    continue
                index_to_topics[index] = topic
                index += 1
        return index_to_topics

    @staticmethod
    def extract_topic_wise_matrices(number_of_topics, document_matrices, type):
        topic_wise_matrices = []
        index = 0
        for i in range(number_of_topics):
            topic_wise_matrix = document_matrices[Utility.Types[type]][index : index + Utility.document_set_sizes[type]]
            if type != "Train":
                topic_wise_matrix = (topic_wise_matrix > 0).astype('float')
            topic_wise_matrices.append(topic_wise_matrix)
            index += Utility.document_set_sizes[type]
        return numpy.asarray(topic_wise_matrices)

class KNN:
    def __init__(self, k):
        self.k = k

    def calculate_distance(self, train_matrix, test_matrix, heuristic):
        if heuristic == "hamming":
            heuristic = "euclidean"
        return pairwise_distances(train_matrix, test_matrix, metric = heuristic)
       

    def predict(self, X_train, Y_train, X_test, heuristic):
        distance_matrix = self.calculate_distance(X_train, X_test, heuristic)

        predicted_labels = []
        
        for sample_wise_distances in distance_matrix.T:
            sorted_indices = numpy.argpartition(sample_wise_distances, self.k)
            # sorted_distances = sample_wise_distances[sorted_indices[ : self.k]]
            sorted_labels = Y_train[sorted_indices[ : self.k]]
            vote_count = defaultdict(int)
            for label in sorted_labels:
                vote_count[label] += 1  # counting vote of label
            sample_predicted_label = max(vote_count, key = vote_count.get)
            predicted_labels.append(sample_predicted_label)

        return numpy.asarray(predicted_labels)
    

    def performance_evaluation(self, train_matrix, train_labels, test_matrix, test_labels, heuristic):
        if heuristic == "hamming":
            X_train = (train_matrix.copy() > 0).astype('float')
            X_test = (test_matrix.copy() > 0).astype('float')
        elif heuristic == "euclidean":
            X_train = train_matrix.copy()
            X_test = test_matrix.copy()
        else:
            X_train, X_test = Utility.generate_data_for_cosine_similarity(train_matrix, test_matrix)

        Y_train = train_labels
        actual_labels = test_labels

        predicted_labels = self.predict(X_train, Y_train, X_test, heuristic)

        return accuracy_score(actual_labels, predicted_labels) * 100   

class NaiveBayes:
    def __init__(self):
        self.topic_wise_probability_matrices = None  # probability of each word falling under a specific topic

    @staticmethod
    def transform(row, total_words, total_unique_words, smoothing_factor):
        return (row + smoothing_factor) / (total_words + smoothing_factor * total_unique_words)

    def train(self, topic_wise_matrices_train, total_unique_words, smoothing_factor):
        topic_wise_probability = []
        for topic_wise_matrix in topic_wise_matrices_train:
            total_words = numpy.sum(topic_wise_matrix)
            number_of_occurrences = numpy.sum(topic_wise_matrix, axis = 0, keepdims = True)
            probability = numpy.apply_along_axis(NaiveBayes.transform, 0, number_of_occurrences, total_words, total_unique_words, smoothing_factor)
            topic_wise_probability.append(probability)
        self.topic_wise_probability_matrices = numpy.concatenate((topic_wise_probability))
    
    def predict(self, index_to_topics, test_document, prior_probabilities):
        probabilities = self.topic_wise_probability_matrices * test_document
        probabilities = minmax_scale(probabilities, feature_range = (1,2))
        probabilities = numpy.prod(probabilities, axis = 1) * prior_probabilities
        return index_to_topics[numpy.argmax(probabilities)]

    def performance_evaluation(self, index_to_topics, topic_wise_test_matrices, type, test_iterations = 50):
        total_documents = numpy.sum([len(topic_wise_test_matrix) for topic_wise_test_matrix in topic_wise_test_matrices])
        prior_probabilities = numpy.asarray([[len(topic_wise_test_matrix) / total_documents  for topic_wise_test_matrix in topic_wise_test_matrices]]).T
        correctly_predicted = 0
        for topic, topic_wise_test_matrix in zip(index_to_topics.values(), topic_wise_test_matrices):
            for document in topic_wise_test_matrix:
                predicted_topic = self.predict(index_to_topics, document, prior_probabilities)
                if predicted_topic == topic:
                    correctly_predicted += 1

        total_samples = len(index_to_topics) * Utility.document_set_sizes[type]
        if type == "Test":
            total_samples /= test_iterations
        return correctly_predicted / total_samples * 100
    

def main():
    Utility.manage_dependencies()

    all_unique_words = Utility.generate_all_unique_words()
    document_matrices, label_matrices = Utility.generate_all_documents(all_unique_words)
    
    index_to_topics = Utility.get_topics()
    
    train_matrix = document_matrices[Utility.Types["Train"]]
    validation_matrix = document_matrices[Utility.Types["Validation"]]
    test_matrix = document_matrices[Utility.Types["Test"]]
    
    train_labels = label_matrices[Utility.Types["Train"]]
    validation_labels = label_matrices[Utility.Types["Validation"]]
    test_labels = label_matrices[Utility.Types["Test"]]

    k_values = [1, 3, 5]
    heuristics = ["hamming", "euclidean", "cosine"]
    for heuristic in heuristics:
        for k in k_values:
            knn = KNN(k = k)
            accuracy = knn.performance_evaluation(train_matrix, train_labels, validation_matrix, validation_labels, heuristic)
            print("KNN Algorithm: accuracy for k =", k, "and heuristic =", heuristic, ":", accuracy)

    topic_wise_matrices_train = Utility.extract_topic_wise_matrices(len(index_to_topics), document_matrices, "Train")
    topic_wise_matrices_validation = Utility.extract_topic_wise_matrices(len(index_to_topics), document_matrices, "Validation")
    topic_wise_matrices_test = Utility.extract_topic_wise_matrices(len(index_to_topics), document_matrices, "Test")
    
    start_smoothing_factor = 0.1
    end_smoothing_factor = 1
    count = 10

    for smoothing_factor in numpy.linspace(start_smoothing_factor, end_smoothing_factor, count):
        nb = NaiveBayes()
        nb.train(topic_wise_matrices_train, len(all_unique_words), smoothing_factor)
        accuracy = nb.performance_evaluation(index_to_topics, topic_wise_matrices_validation, "Validation")
        print("Naive Bayes Algorithm: smoothing factor=", smoothing_factor, " accuracy:", accuracy)

    best_k = 5
    best_heuristic = "cosine"
    best_smoothing_factor = 0.1

    knn = KNN(k = best_k)

    nb = NaiveBayes()
    nb.train(topic_wise_matrices_train, len(all_unique_words), best_smoothing_factor)

    iterations = 50
    per_iteration_documents = 10

    knn_accuracy_values = []
    nb_accuracy_values = []

    for i in range(iterations):
        documents_list = []
        labels_list = []
        for topic, topic_wise_matrix in zip(index_to_topics.values(), topic_wise_matrices_test):
            start = i * per_iteration_documents
            end = start + per_iteration_documents
            documents_list.append(topic_wise_matrix[start : end])
            labels_list += ([topic] * per_iteration_documents)

        knn_matrix = numpy.concatenate((documents_list))
        nb_matrix = numpy.asarray(documents_list)
        label_matrix = numpy.hstack((labels_list))

        knn_accuracy = knn.performance_evaluation(train_matrix, train_labels, knn_matrix, label_matrix, best_heuristic)
        nb_accuracy = nb.performance_evaluation(index_to_topics, nb_matrix, "Test")

        knn_accuracy_values.append(knn_accuracy)
        nb_accuracy_values.append(nb_accuracy)

        print("Iteration = ", i + 1, "KNN Accuracy: ", knn_accuracy, "NB Accuracy: ", nb_accuracy)

    knn_average_accuracy = numpy.sum(knn_accuracy_values) / len(knn_accuracy_values)
    nb_average_accuracy = numpy.sum(nb_accuracy_values) / len(nb_accuracy_values)
    
    print("KNN Average Accuracy:", knn_average_accuracy)
    print("Naive Bayes Average Accuracy:", nb_average_accuracy)

    stat, p_value = ttest_rel(knn_accuracy_values, nb_accuracy_values)
    print(stat, p_value)


if __name__ == "__main__":
    main()
