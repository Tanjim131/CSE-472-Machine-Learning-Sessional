import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from scipy import stats
import math
import sys
# np.set_printoptions(threshold=sys.maxsize)


def impute_numerical(data, column_name, missing_value_indicator):
    data.replace({missing_value_indicator: np.nan}, inplace=True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    return imputer.fit_transform(data[column_name].values.reshape(-1, 1)) 

def impute_categorical(data, column_name, missing_value_indicator):
    data.replace({missing_value_indicator: np.nan}, inplace=True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    return imputer.fit_transform(data[column_name].values.reshape(-1, 1)) 

def calculate_IQR(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    if(IQR == 0):
        IQR = stats.median_absolute_deviation(data[column_name].values, scale=1)
    return IQR

def freedman_diaconis(data, column_name):
    column_IQR = calculate_IQR(data, column_name)
    num_observations = len(data)
    column_bin_width = 2 * column_IQR * (num_observations ** (-1 / 3))
    column_values = data[column_name].values
    column_bins = math.ceil((np.max(column_values) - np.min(column_values)) / column_bin_width)
    return column_bins

def discretization(data, column_name):
    column_bins = freedman_diaconis(data, column_name)
    column_discretizer = KBinsDiscretizer(n_bins=column_bins, encode='ordinal', strategy='quantile')
    return column_discretizer.fit_transform(data[column_name].values.reshape(-1, 1)).astype(np.int32)

def calculate_gini_index(left_node, right_node):
    lnp, lnn = left_node
    rnp, rnn = right_node
    
    left_node_total = lnp + lnn
    right_node_total = rnp + rnn
    
    left_node_gini = 1 - (lnp/left_node_total)**2 - (lnn/left_node_total)**2
    right_node_gini = 1 - (rnp/right_node_total)**2 - (rnn/right_node_total)**2
    
    total_gini = (left_node_total * left_node_gini + right_node_total * right_node_gini)/(left_node_total + right_node_total)
    
    return total_gini

def get_threshold_value(data, column_name, target_feature):
    MAX_SPLIT_POINTS = 1000
    
    sorted_values = np.unique(data[column_name])
    possible_splitting_values = []
    
    if len(sorted_values) > MAX_SPLIT_POINTS:
        indexes = range(0, len(sorted_values) - 1, math.ceil(len(sorted_values) / MAX_SPLIT_POINTS))
    else:
        indexes = range(0, len(sorted_values) - 1)
    
    for i in indexes:
        possible_splitting_values.append((sorted_values[i] + sorted_values[i + 1])/2)
        
    minimum_gini_index = 999 # start with maximum value
    best_split_value = None
        
    for split_value in possible_splitting_values:
        left_node_positive = len(data[(data[column_name] <= split_value) & (data[target_feature] == 1)])
        left_node_negative = len(data[(data[column_name] <= split_value) & (data[target_feature] == 0)])
        
        right_node_positive = len(data[(data[column_name] > split_value) & (data[target_feature] == 1)])
        right_node_negative = len(data[(data[column_name] > split_value) & (data[target_feature] == 0)])
        
        current_gini_index = calculate_gini_index([left_node_positive, left_node_negative], [right_node_positive, right_node_negative])
        
        if minimum_gini_index > current_gini_index:
            minimum_gini_index = current_gini_index
            best_split_value = split_value
            
    return best_split_value

def binarization(data, column_name, target_feature):
    threshold_value = get_threshold_value(data, column_name, target_feature)
    column_binarizer = Binarizer(threshold=threshold_value,copy=False)
    return column_binarizer.fit_transform(data[column_name].values.reshape(-1,1))

def entropy(data, feature):
    elements, counts = np.unique(data[feature], return_counts=True)
    # print(len(data[feature]), np.sum(counts))
    feature_entropy = 0
    for i in range(len(elements)):
        term = counts[i] / len(data[feature])
        feature_entropy += - term * np.log2(term)
    return feature_entropy


def information_gain(data, split_feature, target_feature):
    total_entropy = entropy(data, target_feature)

    elements, counts = np.unique(data[split_feature], return_counts=True)

    split_feature_entropy = 0
    for i in range(len(elements)):
        split_feature_entropy += (counts[i] / len(data[split_feature])) * \
                                 entropy(data.where(data[split_feature] == elements[i]).dropna(), target_feature)
    return total_entropy - split_feature_entropy


def plurality_value(data, target_feature):
    return np.unique(data[target_feature])[np.argmax(np.unique(data[target_feature], return_counts=True)[1])]


def decision_tree_learning(data, features, target_feature, MAX_DEPTH = 20, parent_data = None , depth = 0):
    if len(data) == 0:
        # if data is empty then return max occurrence of target_feature in parent_data
        return plurality_value(parent_data, target_feature)
    elif len(np.unique(data[target_feature])) == 1:
        # if all examples have same classification, return the classification
        return np.unique(data[target_feature])[0]
    elif len(features) == 0 or depth == MAX_DEPTH:
        # if features is empty, then return max occurrence of target_feature in data
        return plurality_value(data, target_feature)
    else:
        # none of the above conditions were hit, recursively grow the tree
        # for every feature, calculate information gain
        information_gain_values = [information_gain(data, feature, target_feature) for feature in features]
        best_feature_index = np.argmax(information_gain_values)
        best_feature = features[best_feature_index]
        # print(best_feature)

        # Create the tree structure.
        # The root gets the name of the feature (best_feature) with the maximum information
        decision_tree = {best_feature: {}}

        # Remove the best feature
        updated_features = np.delete(features, np.where(features == best_feature))
        # print(features)

        # Grow a branch under the root node for each possible value of the root node feature

        for feature_value in np.unique(data[best_feature]):
            # print(feature_value)
            sub_data = data.where(data[best_feature] == feature_value).dropna()
            sub_tree = decision_tree_learning(sub_data, updated_features, target_feature, MAX_DEPTH, data, depth + 1)
            decision_tree[best_feature][feature_value] = sub_tree
        
        decision_tree[best_feature]['Value Not Found'] = plurality_value(data, target_feature)

        return decision_tree
    
    
def predict(test_query, decision_tree):
    for key in test_query.keys():
        if key in decision_tree.keys():
            try:
                sub_tree = decision_tree[key][test_query[key]]
            except:
                return decision_tree[key]['Value Not Found']
            
            sub_tree = decision_tree[key][test_query[key]]
            
            if isinstance(sub_tree, dict):
                # an intermediate node
                return predict(test_query, sub_tree)
            else: 
                # a leaf node
                return sub_tree
    
def make_tree(train_dataset, target_feature, MAX_DEPTH):
    return decision_tree_learning(train_dataset, np.array(train_dataset.columns[:-1]), target_feature, MAX_DEPTH) 


def split_dataset(data):
    return train_test_split(data, test_size=0.2, random_state=1505082, shuffle=True)


def dt_predictions(data, target_feature, tree):
    queries = data.iloc[:,:-1].to_dict(orient='records')
    predictions = []
    for i in range(len(data)):
        predictions.append(predict(queries[i], tree))
    return np.array(predictions)


def get_label_counts(labels):
    unique_labels = np.unique(labels)
    bins = unique_labels.searchsorted(labels)
    return np.bincount(bins)
    
def calculate_metrics_dt(labels, predictions):
    counts = get_label_counts(labels)
    real_negatives, real_positives = counts
    
    true_positives = len(np.where((labels == 1) & (predictions == 1))[0])
    false_positives = len(np.where((labels == 0) & (predictions == 1))[0])
    
    true_negatives = len(np.where((labels == 0) & (predictions == 0))[0])
    false_negatives = len(np.where((labels == 1) & (predictions == 0))[0])
    
    TPR = true_positives/real_positives
    TNR = true_negatives/real_negatives
    
    PPV = true_positives/(true_positives + false_positives)
    FDR = 1 - PPV
    
    F1 = 2 * PPV * TPR / (PPV + TPR)
    
    return TPR, TNR, PPV, FDR, F1

def test_dt(data, target_feature, tree):
    # 1 - positive, 0 - negative
    predictions = dt_predictions(data, target_feature, tree)
    accuracy = metrics.accuracy_score(data[target_feature], predictions)
    TPR, TNR, PPV, FDR, F1 = calculate_metrics_dt(data[target_feature].values, predictions)
    report_metrics = (accuracy, TPR, TNR, PPV, FDR, F1) 
    report_metrics = (metric * 100 for metric in report_metrics)
    return report_metrics

#################################################################################################
    
def Resample(data, weight_function, random_state):
    return data.sample(frac=1, replace=True, weights=weight_function, random_state=random_state)


def adaboost(data, features, target_feature, l_weak, k):
    random_state = np.random.RandomState(1505082)
    weights = np.full(len(data), 1/len(data))
    h = []
    z = []
    for i in range(k):
        resampled_data = Resample(data, weights, random_state)
        
        error = 0
        
        decision_stump = l_weak(resampled_data, features, target_feature, 1)
        decision_stump_predictions = dt_predictions(data, target_feature, decision_stump)
        
        query_index = 0
    
        for index, row in data.iterrows():
            if (decision_stump_predictions[query_index] != data.at[index, target_feature]):
                error = error + weights[query_index]
            query_index = query_index + 1

        if error > 0.5:
            continue
            
        h.append(decision_stump)
        
        query_index = 0
        for index, row in data.iterrows():
            if (decision_stump_predictions[query_index] == data.at[index, target_feature]):
                weights[query_index] = weights[query_index] * (error / (1 - error))
            query_index = query_index + 1
            
        weights = weights / np.sum(weights)
        
        z.append(np.log2((1 - error) / error))
    
    return h,z


def adaboost_predictions(data, target_feature, hypotheses, hypotheses_weights):
    decisions = []
    queries = data.iloc[:,:-1].to_dict(orient='records')
    count = 0
    for i in range(len(data)):
        decision = 0
        for j in range(len(hypotheses)):
            prediction = predict(queries[i], hypotheses[j])
            if prediction == 1:
                decision = decision + hypotheses_weights[j]
            else:
                decision = decision - hypotheses_weights[j]
        
        decisions.append(decision)
    
    return decisions


def test_adaboost(data, target_feature, hypotheses, hypotheses_weights):
    decisions = adaboost_predictions(data, target_feature, hypotheses, hypotheses_weights)
    
    count = 0
    for i in range(len(data)):            
        if (decisions[i] > 0 and data[target_feature].iloc[i] == 1) or (decisions[i] < 0 and data[target_feature].iloc[i] == 0):
            count = count + 1
    accuracy = (count / len(data)) * 100
    
    return accuracy


#########################################################################

def preprocess_telco():
    data = pd.read_csv("telco.csv")
    telco_target_feature = 'Churn'
    
    labelencoder = LabelEncoder()
    data[telco_target_feature] = labelencoder.fit_transform(data[telco_target_feature])
    
    data.drop('customerID', axis=1, inplace=True)

    data['TotalCharges'] = impute_numerical(data, 'TotalCharges', ' ')
    
    data['TotalCharges'] = discretization(data, 'TotalCharges')
    data['MonthlyCharges'] = discretization(data, 'MonthlyCharges')
    data['tenure'] = discretization(data, 'tenure')
    
    return data


def preprocess_adult(file_type):
    adult_features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    adult_target_feature = 'label'
    
    data = None
    if(file_type == 'Train'):
        data = pd.read_csv("adult.data", names=adult_features)
    else:
        data = pd.read_csv("adult.test", skiprows = 1, names=adult_features)
    
    data_obj = data.select_dtypes(['object'])
    data[data_obj.columns] = data_obj.apply(lambda x: x.str.strip())
    
    labelencoder = LabelEncoder()
    data[adult_target_feature] = labelencoder.fit_transform(data[adult_target_feature])
    
    data['workclass'] = impute_categorical(data, 'workclass', '?')
    data['occupation'] = impute_categorical(data, 'occupation', '?')
    data['native-country'] = impute_categorical(data, 'native-country', '?')
    
    data['age'] = binarization(data, 'age', adult_target_feature)
    data['fnlwgt'] = binarization(data, 'fnlwgt', adult_target_feature)
    data['education-num'] = binarization(data, 'education-num', adult_target_feature)
    data['capital-gain'] = binarization(data, 'capital-gain', adult_target_feature)
    data['capital-loss'] = binarization(data, 'capital-loss', adult_target_feature)
    data['hours-per-week'] = binarization(data, 'hours-per-week', adult_target_feature)

    return data


def preprocess_creditcard():
    data = pd.read_csv("creditcard.csv")
    creditcard_target_feature = 'Class'
    
    data.drop('Time', axis=1, inplace=True)
    
    for column in data:
        if column != creditcard_target_feature:
            data[column] = binarization(data, column, creditcard_target_feature)
    
    return data


def report_generation(dataset_name, target_feature):
    dataset_df = None
    
    train_df = None
    test_df = None
    
    if(dataset_name == 'Telco'):
        dataset_df = preprocess_telco()
        train_df, test_df = split_dataset(dataset_df)  
    elif(dataset_name == 'Adult'):
        train_df = preprocess_adult('Train')
        test_df = preprocess_adult('Test')
    else:
        dataset_df = preprocess_creditcard()
        train_df, test_df = split_dataset(dataset_df)

    decision_tree = make_tree(train_df, target_feature, len(train_df.columns))
    h_5, z_5 = adaboost(train_df, train_df.columns[:-1], target_feature, decision_tree_learning, 5)
    h_10, z_10 = adaboost(train_df, train_df.columns[:-1], target_feature, decision_tree_learning, 10)
    h_15, z_15 = adaboost(train_df, train_df.columns[:-1], target_feature, decision_tree_learning, 15)
    h_20, z_20 = adaboost(train_df, train_df.columns[:-1], target_feature, decision_tree_learning, 20)
    
    print("------------------------ Decision Tree ------------------------ ")
    
    train_accuracy_dt, train_tpr_dt, train_tnr_dt, train_ppv_dt, train_fdr_dt, train_f1_dt = test_dt(train_df, target_feature, decision_tree)
    print(dataset_name + "_train_accuracy: ", train_accuracy_dt)
    print(dataset_name + "_train_tpr: ", train_tpr_dt)
    print(dataset_name + "_train_tnr: ", train_tnr_dt)
    print(dataset_name + "_train_ppv: ", train_ppv_dt)
    print(dataset_name + "_train_fdr: ", train_fdr_dt)
    print(dataset_name + "_train_f1: ", train_f1_dt)
    
    print()
    
    test_accuracy_dt, test_tpr_dt, test_tnr_dt, test_ppv_dt, test_fdr_dt, test_f1_dt = test_dt(test_df, target_feature, decision_tree)
    print(dataset_name + "_test_accuracy: ", test_accuracy_dt)
    print(dataset_name + "_test_tpr: ", test_tpr_dt)
    print(dataset_name + "_test_tnr: ", test_tnr_dt)
    print(dataset_name + "_test_ppv: ", test_ppv_dt)
    print(dataset_name + "_test_fdr: ", test_fdr_dt)
    print(dataset_name + "_test_f1: ", test_f1_dt)
    
    print()
    
    print("------------------------  AdaBoost ------------------------ ")
    
    print("Number of boosting rounds, k = 5")
    
    train_accuracy_adaboostk5 =  test_adaboost(train_df, target_feature, h_5, z_5)
    print(dataset_name + " train_accuracy: ", train_accuracy_adaboostk5)
    
    test_accuracy_adaboostk5 = test_adaboost(test_df, target_feature, h_5, z_5)
    print(dataset_name + " test_accuracy: ", test_accuracy_adaboostk5)
    
    print()
    print("Number of boosting rounds, k = 10")
    
    train_accuracy_adaboostk10 =  test_adaboost(train_df, target_feature, h_10, z_10)
    print(dataset_name + " train_accuracy: ", train_accuracy_adaboostk10)

    test_accuracy_adaboostk10 = test_adaboost(test_df, target_feature, h_10, z_10)
    print(dataset_name + " test_accuracy: ", test_accuracy_adaboostk10)

    print()
    print("Number of boosting rounds, k = 15")

    train_accuracy_adaboostk15 =  test_adaboost(train_df, target_feature, h_15, z_15)
    print(dataset_name + " train_accuracy: ", train_accuracy_adaboostk15)

    test_accuracy_adaboostk15 = test_adaboost(test_df, target_feature, h_15, z_15)
    print(dataset_name + " test_accuracy: ", test_accuracy_adaboostk15)

    print()
    print("Number of boosting rounds, k = 20")
    
    train_accuracy_adaboostk20 =  test_adaboost(train_df, target_feature, h_20, z_20)
    print(dataset_name + " train_accuracy: ", train_accuracy_adaboostk20)

    test_accuracy_adaboostk20 = test_adaboost(test_df, target_feature, h_20, z_20)
    print(dataset_name + " test_accuracy: ", test_accuracy_adaboostk20)
    
    print()
    
    
report_generation('Telco', 'Churn')
report_generation('Adult', 'label')
report_generation('Credit Card', 'Class')