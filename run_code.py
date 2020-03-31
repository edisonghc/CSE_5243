import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords as sw
STOP_WORDS = set(sw.words('english')) 

import string
import copy
import os
import sys
import operator
import time
import numpy as np
from random import seed
from random import randrange


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds): 
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Stratified cross-validation split
def stratified_datasplit(root_path):
    print(f'|   Starting stratified_datasplit')
    start_time = time.time()

    # Only keep the data dictionaries and ignore possible system files like .DS_Store
    folders = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    folds = [[] for i in range(N_FOLDS)]
    i = 0
    for folder in folders:
        print(f'|   |   {i+1}) Splitting data from {folder}')
        files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
        folds = np.column_stack((cross_validation_split(files, N_FOLDS),folds))
        i += 1
    
    elapsed_time = time.time() - start_time
    print(f'|   stratified_datasplit took {elapsed_time:.2f} seconds')
    print()
    return folds


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(root_path, algorithm, *args):
    print(f'Evaluating {algorithm.__name__}')
    start_time = time.time()

    folds = stratified_datasplit(root_path)
    scores = list()
    count_cv = 0
    for i in range(len(folds)):

        # A switch to run the algorithm just once for efficiency
        # if i > 0:
        #   continue

        print(f'|   {i+1}th iteration')
        train_set = list()
        for j in range(len(folds)):
            if not i == j:
                train_set = np.append(train_set, copy.deepcopy(folds[j]))
        print(f'|   |   Building feature matrix on training data')
        train_vocab = construct_vocab(train_set,count_cv)
        count_cv += 1
        train_data, label2id = extract_feature(train_set, train_vocab)
        print()
        test_set = list(folds[i])
        print(f'|   |   Building feature matrix on testing data')
        test_data_with_label, _ = extract_feature(test_set, train_vocab, label2id)
        test_data_no_label = test_data_with_label[:,:-1]
        print()
        print(f'|   |   Running {algorithm.__name__}')
        predicted = algorithm(train_data, test_data_no_label, *args)
        print()
        print(f'|   |   Caculating the accuracy')
        actual = [row[-1] for row in test_data_with_label]
        accuracy = accuracy_metric(actual, predicted)
        print(f'|   |   Accuracy is {accuracy:.2f}%')
        scores.append(accuracy)
        print()
    elapsed_time = time.time() - start_time
    print(f'Evaluating {algorithm.__name__} took {elapsed_time/60:.2f} minutes')
    print()
    return scores


# Vocabulary Construction
def construct_vocab(files,count_cv):
    print(f'|   |   |   Constructing vocabulary')
    start_time = time.time()

    vocab_full = {}
    n_doc = 0
    for file in files:
        n_doc += 1
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                # split into words
                tokens = word_tokenize(line)
                # convert to lower case
                tokens = [w.lower() for w in tokens]
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words = [word for word in stripped if word.isalpha()]
                # filter out stop words
                words = [w for w in words if not w in STOP_WORDS]
                # stemming of words
                porter = PorterStemmer()
                stemmed = [porter.stem(word) for word in words]
                for token in stemmed:
                    vocab_full[token] = vocab_full.get(token, 0) + 1
    print(f'|   |   |   |   {n_doc} documents scaned')
    print(f'|   |   |   |   Full vocabulary has {len(vocab_full)} words')
    vocab_sorted = sorted(vocab_full.items(), key=operator.itemgetter(1), reverse=True)
    ideal_vocab_size = min(len(vocab_sorted),MAX_VOCAB_SIZE)
    vocab_truncated = vocab_sorted[:ideal_vocab_size]
    # Save the vocabulary to file for visual inspection and possible analysis
    with open(f'vocab_{count_cv+1}.txt', 'w') as f:
        for vocab, freq in vocab_truncated:
            f.write(f'{vocab}\t{freq}\n')
    # The final vocabulary is a dict mapping each token to its id. frequency information is not needed anymore.
    vocab = dict([(token, id) for id, (token, _) in enumerate(vocab_truncated)])
    # Since we have truncated the vocabulary, we will encounter many tokens that are not in the vocabulary. We will map all of them to the same 'UNK' token (a common practice in text processing), so we append it to the end of the vocabulary.
    vocab['UNK'] = ideal_vocab_size

    print(f'|   |   |   |   Truncated vocabulary has {len(vocab)} words')
    elapsed_time = time.time() - start_time
    print(f'|   |   |   Constructing vocabulary took {elapsed_time/60:.2f} minutes')
    return vocab


def extract_feature(files, vocab, label2id = 0):
    print(f'|   |   |   Extracting feature')
    start_time = time.time()

    # Since we have truncated the vocabulary, it's now reasonable to hold the entire feature matrix in memory (it takes about 3.6GB on a 64-bit machine). If memory is an issue, you could make the vocabulary even smaller or use sparse matrix.
    features = np.zeros((len(files), len(vocab)), dtype=int)
    print(f'|   |   |   |   Feature matrix takes {sys.getsizeof(features)/1000000:.4f} Mb')

    # The class label of each document
    labels = np.zeros(len(files), dtype=int)
    # The mapping from the name of each class label (i.e., the subdictionary name corresponding to a topic) to an integer ID
    doc_id = 0
    if label2id == 0:
        folders = list(set(os.path.dirname(file) for file in files))
        label2id = dict([(label, id) for id, label in enumerate(folders)])
    for file in files:
        label = os.path.dirname(file)
        labels[doc_id] = label2id[label]
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                # split into words
                tokens = word_tokenize(line)
                # convert to lower case
                tokens = [w.lower() for w in tokens]
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                stripped = [w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                words = [word for word in stripped if word.isalpha()]
                # filter out stop words
                words = [w for w in words if not w in STOP_WORDS]
                # stemming of words
                porter = PorterStemmer()
                stemmed = [porter.stem(word) for word in words]
                for token in stemmed:
                    # if the current token is in the vocabulary, get its ID; otherwise, get the ID of the UNK token
                    unk_id = len(vocab) - 1
                    token_id = vocab.get(token, unk_id)
                    features[doc_id, token_id] += 1
        doc_id += 1

    # id2label = dict([(id, label) for label, id in label2id.items()])
    dataset = np.column_stack((features, labels))
    print(f'|   |   |   |   Dataset has dimension {dataset.shape}')
    elapsed_time = time.time() - start_time
    print(f'|   |   |   Extracting feature took {elapsed_time/60:.2f} minutes')
    return dataset, label2id


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        tmp = [row[-1] for row in group]
        for class_val in classes:
            p = tmp.count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    start_time = time.time()
    print(f'|   |   |   |   Finding a split')
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 0, 0, 1, None
    for index in range(len(dataset[0])-1):
        iteration_start_time = time.time()
        # print(f'|   |   |   |   |   {index+1}th attribute')
        left = list()
        tmp = [[row[index],row[-1]] for row in list(dataset)]
        right = sorted(tmp, key = operator.itemgetter(0), reverse = True)
        left.append(right.pop())
        while (len(right) > MIN_SIZE) and ((len(left) < MIN_SIZE) or (right[-1][0] == left[-1][0])):
            left.append(right.pop())
        updated = False
        length = len(right)
        for i in range(length):
            if (len(right) <= MIN_SIZE) or (right[0][0] == 0):
                break
            groups = left, right
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score = index, right[-1][0]-0.5, gini
                updated = True
                b_i = i
            left.append(right.pop())
            while (len(right) > MIN_SIZE) and (right[-1][0] == left[-1][0]) :
                left.append(right.pop())
        iteration_time = time.time() - iteration_start_time
        ## if updated:
            ## print(f"|   |   |   |   |   Best Gini = {b_score:.6f} at {index+1}th attribute {str(b_i)}th row")
        # print(f"|   |   |   |   |   |   Spent {iteration_time:.0f} seconds, Best Gini = {b_score:.6f} at {index+1}th attribute{(' '+str(b_i)+'th row') if updated else ', did not change'}")
    b_groups = test_split(b_index,b_value,dataset)
    n_rows = [len(group) for group in b_groups]
    acc = to_terminal(dataset)
    elapsed_time = time.time() - start_time
    print(f'|   |   |   |   Finding a split took {elapsed_time:.2f} seconds')
    return {'index':b_index, 'value':b_value, 'groups':b_groups, 'left_rows': n_rows[0], 'right_rows': n_rows[1], 'accuracy': acc}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    tmp = [max(set(outcomes), key=outcomes.count)]
    predicted = tmp * len(group)
    accuracy = accuracy_metric(outcomes, predicted)
    return [tmp[0], accuracy, len(group)]


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    # print(f"{(depth-1)*'  :  '}[X{node['index']+1} < {node['value']}]") 
    left, right = node['groups']
    del(node['groups'])
    
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        # print(f"{depth*'  :  '}single root ... [{node['left']}]")
        return True
    
    # check for max depth
    if depth >= max_depth:
        node['left'] = to_terminal(left)
        # print(f"{depth*'  :  '}max_depth ... [{node['left']}]")
        node['right'] = to_terminal(right)
        # print(f"{depth*'  :  '}max_depth ... [{node['right']}]")
        if node['left'][0] == node['right'][0]:
            node['left'] = node['right'] = to_terminal(left + right)
            return True
        else:
            return False
    
    # process left child
    min_left = False
    single_left = False
    if len(left) <= 2*min_size:
        node['left'] = to_terminal(left)
        min_left = True
        # print(f"{depth*'  :  '}min_size on left... [{node['left']}]")
    else:
        node['left'] = get_split(left)
        # if min(node['left']['left_rows'],node['left']['right_rows']) <= min_size:
        #     node['left'] = to_terminal(left)
        #     min_left = True
        #     # print(f"{(depth+1)*'  :  '}single right ... [{node['right']}]")
        # else:
        single_left = split(node['left'], max_depth, min_size, depth+1)
        if single_left:
            node['left'] = to_terminal(left)
            # print(f"{(depth+1)*'  :  '}single left ... [{node['left']}]")
    left_isleaf = min_left or single_left
    
    # process right child
    min_right = False
    single_right = False
    if len(right) <= 2*min_size:
        node['right'] = to_terminal(right)
        min_right = True
        # print(f"{depth*'  :  '}min_size on right ... [{node['right']}]")
    else:
        node['right'] = get_split(right)
        # if min(node['right']['left_rows'],node['right']['right_rows']) <= min_size:
        #     node['right'] = to_terminal(right)
        #     min_right = True
        #     # print(f"{(depth+1)*'  :  '}single right ... [{node['right']}]")
        # else:
        single_right = split(node['right'], max_depth, min_size, depth+1)
        if single_right:
            node['right'] = to_terminal(right)
            # print(f"{(depth+1)*'  :  '}single right ... [{node['right']}]")
    right_isleaf = min_right or single_right
    
    if left_isleaf and right_isleaf and node['left'][0] == node['right'][0]:
        node['left'] = node['right'] = to_terminal(left + right)
        return True
    else:
        return False


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    single = False
    if root['value'] == 0:
        root = to_terminal(train)
    else:
        single = split(root, max_depth, min_size, 1)
        if single:
            root = to_terminal(train)
    return root


# Print a decision tree
def print_tree(node, depth=1):
    if isinstance(node, dict):
        print(f"{(depth-1)*'  │  '}  ├──[X{node['index']+1} < {node['value']}] . . . . . . {node['accuracy'][1]:.2f}% of {node['accuracy'][2]} rows")
        print_tree(node['left'], depth+1)
        print_tree_right(node['right'], depth+1)
    else:
        print(f"{(depth-1)*'  │  '}  ├──[{node[0]}] . . . . . . {node[1]:.2f}% of {node[2]} rows")

# Print a decision tree
def print_tree_right(node, depth):
    if isinstance(node, dict):
        print(f"{(depth-1)*'  │  '}  └──[X{node['index']+1} < {node['value']}] . . . . . . {node['accuracy'][1]:.2f}% of {node['accuracy'][2]} rows")
        print_tree(node['left'], depth+1)
        print_tree_right(node['right'], depth+1)
    else:
        print(f"{(depth-1)*'  │  '}  └──[{node[0]}] . . . . . . {node[1]:.2f}% of {node[2]} rows")


# Make a prediction with a decision tree
def predict(node, row):
    if not isinstance(node, dict):
        return node[0]
    else: 
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left'][0]
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right'][0]
 

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    start_time = time.time()
    print(f'|   |   |   Building Decision Tree')
    tree = build_tree(train, max_depth, min_size)
    build_time = time.time() - start_time
    print(f'|   |   |   Building Decision Tree took {build_time/60:.2f} minutes')
    print()
    print_tree(tree)
    print()
    start_time = time.time()
    print(f'|   |   |   Making predictions')
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    predict_time = time.time() - start_time
    print(f'|   |   |   Making predictions took {predict_time:.2f} seconds')
    return predictions


run_start_time = time.time()
# The maximum size of the final vocabulary. It's a hyper-parameter. You can change it to        see what value gives the best performance.
MAX_VOCAB_SIZE = 5000
N_FOLDS = 5 #10

MAX_DEPTH = 50000
MIN_SIZE = 15

# Assuming this file is put under the same parent directoray as the data directory, and the     data directory is named "20news-train"
root_path = "./20news-train"

# Test CART on Bank Note dataset
seed(1)
# evaluate algorithm

scores= evaluate_algorithm(root_path, decision_tree, MAX_DEPTH, MIN_SIZE)

print("Scores:", *(f"{s:.3f}%" for s in scores))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print()
run_time = time.time() - run_start_time
print(f'The program ran for {run_time/60:.2f} minutes')
print('Done!')