import math
import statistics
import numpy as np
import matplotlib.pyplot as plot

############################# supporting functions for Decision Tree ################################
# Load train.csv and test.csv
with open('train.csv') as f:
    training_data = [];
    for line in f:
        terms = line.strip().split(',')
        training_data.append(terms)

# numerical attributes
numerical_attributes_dict = {0: 0, 5: 5, 9: 9, 11: 11, 12: 12, 13: 13, 14: 14}

# function that converts only the numerical strings to floats
def convert_to_float(input_data):
    original_data = input_data
    for i in range(len(original_data)):
        for j in set(numerical_attributes_dict.keys()):
            original_data[i][j] = float(input_data[i][j])
    return original_data

training_data = convert_to_float(training_data)

# obtain the median of the labels
for i in numerical_attributes_dict:
    numerical_attributes_dict[i] = statistics.median([element[i] for element in training_data])

# assign categories
for element in training_data:
    for k in numerical_attributes_dict:
        if element[k] >= numerical_attributes_dict[k]:
            element[k] = 'yes'
        else:
            element[k] = 'no'

# store attributes in a dictionary
given_attributes = {'age': ['yes', 'no'],
             'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                     'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
             'martial': ['married', 'divorced', 'single'],
             'education': ['unknown', 'secondary', 'primary', 'tertiary'],
             'default': ['yes', 'no'],
             'balance': ['yes', 'no'],
             'housing': ['yes', 'no'],
             'loan': ['yes', 'no'],
             'contact': ['unknown', 'telephone', 'cellular'],
             'day': ['yes', 'no'],
             'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
             'duration': ['yes', 'no'],
             'campaign': ['yes', 'no'],
             'pdays': ['yes', 'no'],
             'previous': ['yes', 'no'],
             'poutcome': ['unknown', 'other', 'failure', 'success']}

# convert categorical data to numerical values to facilitate manipulation
def convert_attribute_to_num(attribute):
    numerical_attribute = 0
    if attribute == 'age':
        numerical_attribute = 0
    elif attribute == 'job':
        numerical_attribute = 1
    elif attribute == 'martial':
        numerical_attribute = 2
    elif attribute == 'education':
        numerical_attribute = 3
    elif attribute == 'default':
        numerical_attribute = 4
    elif attribute == 'balance':
        numerical_attribute = 5
    elif attribute == 'housing':
        numerical_attribute = 6
    elif attribute == 'loan':
        numerical_attribute = 7
    elif attribute == 'contact':
        numerical_attribute = 8
    elif attribute == 'day':
        numerical_attribute = 9
    elif attribute == 'month':
        numerical_attribute = 10
    elif attribute == 'duration':
        numerical_attribute = 11
    elif attribute == 'campaign':
        numerical_attribute = 12
    elif attribute == 'pdays':
        numerical_attribute = 13
    elif attribute == 'previous':
        numerical_attribute = 14
    elif attribute == 'poutcome':
        numerical_attribute = 15
    elif attribute == 'y':
        numerical_attribute = 16
    return numerical_attribute

# Create empty dictionary to store attributes
def append_attribute(attribute):
    empty_dict = {}
    for each_attribute in given_attributes[attribute]:
        empty_dict[each_attribute] = []
    return empty_dict

# Empty dictionary for storing information gain values
def information_index_store(attributes):
    empty_dict = {}
    for each_attribute in attributes:
        empty_dict[each_attribute] = 0
    return empty_dict

# function that calculates the entropy to find IG
def calculate_entropy(groups, classes):
    # N_ins= float(sum([len(groups[attr_val]) for attr_val in groups]))
    Q = 0.0  # total weights
    tp = 0.0
    for attr_val in groups:
        tp = sum([row[-1] for row in groups[attr_val]])
        Q = Q + tp
    exp_ent = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue  # jump this iteration
        score = 0
        q = sum([row[-1] for row in groups[attr_val]])
        for class_val in classes:
            #           p = [row[-3] for row in groups[attr_val]].count(class_val) / size   ###
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val]) / q  # sum up the weights
            if p == 0:
                temp = 0
            else:
                temp = p * math.log2(1 / p)
            score += temp
        #        exp_ent += score* (size / N_ins)
        exp_ent += score * sum([row[-1] for row in groups[attr_val]]) / Q  # total weights of a subset
    return exp_ent
# '''

# function that divides data based on labels
def divide_data(attribute,input_data):
    child = append_attribute(attribute)
    for ith_element in input_data:
        for this_attribute in given_attributes[attribute]:
            numerical_attribute = convert_attribute_to_num(attribute)
            if ith_element[numerical_attribute] == this_attribute:
                child[this_attribute].append(ith_element)
    return child


def best_splitting_feature(input_data):
    if input_data == []:
        return
    labels = list(set(elem[-2] for elem in input_data))
    info_index_value = information_index_store(given_attributes)
    for attribute in given_attributes:
        batches = divide_data(attribute, input_data)
        info_index_value[attribute] = calculate_entropy(batches, labels)
    best_feature = min(info_index_value, key=info_index_value.get)
    best_subset = divide_data(best_feature, input_data)
    output = {'best_splitting_feature': best_feature, 'best_subset': best_subset}
    return output


# function that assigns most common label to a group
def find_common_label(group):
    labels = []
    for element in group:
        labels.append(element[-2])
    most_common_label = max(set(labels), key= labels.count)
    return most_common_label

# function that recursively splits the data up to a maximum depth
def create_branch(branch, max_tree_depth, tree_depth):
    if tree_depth >= max_tree_depth:
        for label in branch['best_subset']:
            if branch['best_subset'][label] != []:
                branch[label] = find_common_label(branch['best_subset'][label])
            else:
                branch[label] = find_common_label(sum(branch['best_subset'].values(), []))
        return

    for label in branch['best_subset']:
        if branch['best_subset'][label] != []:
            branch[label] = best_splitting_feature(branch['best_subset'][label])
            #tree_depth +=1
            create_branch(branch[label], max_tree_depth, tree_depth + 1)
        else:
            branch[label] = find_common_label(sum(branch['best_subset'].values(), [ ]))

# build decision tree
def Decision_Tree(input_data, max_tree_depth):
    first_layer = best_splitting_feature(input_data)
    create_branch(first_layer, max_tree_depth, 1)
    return first_layer

############################# Supporting functions for prediction error for Adaboost ################################

def check_label(branch, label):
    numeric_label = label[convert_attribute_to_num(branch['best_splitting_feature'])]
    predicted_label = branch[numeric_label]
    if isinstance(predicted_label, dict):
        return check_label(predicted_label, label)
    else:
        return predicted_label

def sgn(value):
    if value > 0:
        return 1.0
    else:
        return -1.0

def get_labels(input_data, tree):
    actual_labels = []
    predicted_labels = []
    for element in input_data:
        actual_labels.append(element[-2])
        prediction = check_label(tree, element)
        predicted_labels.append(prediction)
    return [actual_labels, predicted_labels]

def prediction_list(len):
    empty_dict = {}
    for k in range(len):
        empty_dict[k] = []
    return empty_dict

def convert_to_binary(input_list):
    binary_list = []
    for k in range(len(input_list)):
        if input_list[k] == 'yes':
            binary_list.append(1.0)
        else:
            binary_list.append(-1.0)
    return binary_list


def update_weight(w_current, vote, actual_labels, predicted_labels):  # updating weights
    new_weight = []
    for i in range(len(actual_labels)):
        new_weight.append(w_current[i] * math.e ** (- vote * actual_labels[i] * predicted_labels[i]))
    new_weight = [x / sum(new_weight) for x in new_weight]
    return new_weight

def reupdate_weights(input_list, w):
    for i in range(len(input_list)):
        input_list[i][-1] = w[i]
    return input_list

def store_weights(input_list, w):
    for i in range(len(input_list)):
        input_list[i].append(w[i])
    return input_list

def calculate_final_prediction(stump_prediction, vote, data_len, length):
    prediction_list = []
    for q in range(data_len):
        val = sum([stump_prediction[p][0][q] * vote[p] for p in range(length)])
        prediction_list.append(sgn(val))
    return prediction_list

def calculate_weighted_error(true_labels, predicted_labels, w):
    error_counter = 0
    for i in range(len(true_labels)):
        if true_labels[i] != predicted_labels[i]:
            error_counter += w[i]
    return error_counter

def calculate_error_percentage(actual_labels, predicted_labels):
    inaccurate_predictions = 0
    total_labels = len(actual_labels)
    for i in range(total_labels):
        if actual_labels[i] != predicted_labels[i]:
            inaccurate_predictions += 1
    return inaccurate_predictions / total_labels

################################################ Adaboost Algorithm ################################################

def Adaboost(iter,input_data):
    prediction_per_stump = prediction_list(iter)
    vote_per_stump = []
    w = [this_elem[-1] for this_elem in input_data]
    for i in range(iter):
        all_stumps = Decision_Tree(input_data, 1)
        print('best feature of stump is:',all_stumps['best_splitting_feature'])
        [actual_labels, predicted_labels] = get_labels(input_data, all_stumps)
        prediction_per_stump[i].append(convert_to_binary(predicted_labels))
        weighted_error = calculate_weighted_error(actual_labels, predicted_labels, w)
        print('The error in this stump is',weighted_error)
        print('The current weight ',w[0])
        vote_per_stump.append(0.5 * math.log((1 - weighted_error) / weighted_error))
        w = update_weight(w, 0.5 * math.log((1 - weighted_error) / weighted_error), convert_to_binary(actual_labels), convert_to_binary(predicted_labels))
        input_data = reupdate_weights(input_data, w)
    return [prediction_per_stump, vote_per_stump, w]


empty_weights = np.ones(len(training_data)) / len(training_data)
training_data = store_weights(training_data, empty_weights)
actual_labels_binary = convert_to_binary([elem[-2] for elem in training_data])

# function that calculate the error at each iteration
def calculate_error_in_this_iteration(iter):
    error_list =[]
    for this_iter in range(1,iter):
        print('iteration', this_iter)
        [prediction_per_stump, vote_per_stump, w] = Adaboost(this_iter,training_data)
        final_prediction = calculate_final_prediction(prediction_per_stump, vote_per_stump, len(training_data), this_iter)
        error_list.append(calculate_error_percentage(actual_labels_binary, final_prediction))
    return error_list

################################################## plotting error #################################################
Num_of_iterations = 5
plot.plot(calculate_error_in_this_iteration(Num_of_iterations))
plot.ylabel('Error')
plot.xlabel('Number of iterations')
plot.title('training error vs number of iterations')
plot.show()

