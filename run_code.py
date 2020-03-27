import nltk
nltk.download('punkt')  # needed by word_tokenize
nltk.download('stopwords')
import copy
import os
import sys
import operator
import time
import numpy as np
from random import seed
from random import randrange
from nltk.stem import PorterStemmer as ps
from nltk.corpus import stopwords as sw
STOP_WORDS = set(sw.words('english')) 

from nltk.tokenize import word_tokenize

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
    # Only keep the data dictionaries and ignore possible system files like .DS_Store
    folders = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    folds = [[] for i in range(N_FOLDS)]
    for folder in folders:
        files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
        folds = np.column_stack((cross_validation_split(files, N_FOLDS),folds))
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
    folds = stratified_datasplit(root_path)
    scores = list()
    build_time = list()
    predict_time = list()
    for i in range(len(folds)):
        train_set = list()
        for j in range(len(folds)):
            if not i == j:
                train_set = np.append(train_set, copy.deepcopy(folds[j]))
        train_vocab = construct_vocab(train_set)
        train_data, label2id = extract_feature(train_set, train_vocab)

        test_set = list(folds[i])
        test_data_with_label, _ = extract_feature(test_set, train_vocab, label2id)
        test_data_no_label = test_data_with_label[:,:-1]

        predicted, build_t, predict_t = algorithm(train_data, test_data_no_label, *args)
        actual = [row[-1] for row in test_data_with_label]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        build_time.append(build_t)
        predict_time.append(predict_t)
    return scores, build_time, predict_time

# Vocabulary Construction
def construct_vocab(files):
    ps = nltk.stem.PorterStemmer()
    start_time = time.time()
    vocab_full = {}
    n_doc = 0
    for file in files:
        n_doc += 1
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                tokens = word_tokenize(line)
                filtered_tokens = [w for w in tokens if not w in STOP_WORDS]
                for token in filtered_tokens:
                    root_word = ps.stem(token)
                    vocab_full[root_word] = vocab_full.get(root_word, 0) + 1
    print(f'{n_doc} documents in total with a total vocab size of {len(vocab_full)}')
    vocab_sorted = sorted(vocab_full.items(), key=operator.itemgetter(1), reverse=True)
    ideal_vocab_size = min(len(vocab_sorted),MAX_VOCAB_SIZE)
    vocab_truncated = vocab_sorted[:ideal_vocab_size]
    # Save the vocabulary to file for visual inspection and possible analysis
    with open('vocab.txt', 'w') as f:
        for vocab, freq in vocab_truncated:
            f.write(f'{vocab}\t{freq}\n')
    # The final vocabulary is a dict mapping each token to its id. frequency information is not     needed anymore.
    vocab = dict([(token, id) for id, (token, _) in enumerate(vocab_truncated)])
    # Since we have truncated the vocabulary, we will encounter many tokens that are not in the     vocabulary. We will map all of them to the same 'UNK' token (a common practice in text          processing), so we append it to the end of the vocabulary.
    vocab['UNK'] = ideal_vocab_size
    elapsed_time = time.time() - start_time
    print(f'Vocabulary construction took {elapsed_time} seconds')
    return vocab

def extract_feature(files, vocab, label2id = 0):
    ps = nltk.stem.PorterStemmer()
    # Since we have truncated the vocabulary, it's now reasonable to hold the entire feature        matrix in memory (it takes about 3.6GB on a 64-bit machine). If memory is an issue, you         could make the vocabulary even smaller or use sparse matrix.
    start_time = time.time()
    features = np.zeros((len(files), len(vocab)), dtype=int)
    print(f'The feature matrix takes {sys.getsizeof(features)} Bytes.')
    # The class label of each document
    labels = np.zeros(len(files), dtype=int)
    # The mapping from the name of each class label (i.e., the subdictionary name corresponding     to a topic) to an integer ID
    doc_id = 0
    if label2id == 0:
        folders = list(set(os.path.dirname(file) for file in files))
        label2id = dict([(label, id) for id, label in enumerate(folders)])
    for file in files:
        label = os.path.dirname(file)
        labels[doc_id] = label2id[label]
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                tokens = word_tokenize(line)
                filtered_tokens = [w for w in tokens if not w in STOP_WORDS]
                for token in filtered_tokens:
                    # if the current token is in the vocabulary, get its ID; otherwise, get                         the ID of the UNK token
                    root_word = ps.stem(token)
                    unk_id = len(vocab) - 1
                    token_id = vocab.get(root_word, unk_id)
                    features[doc_id, token_id] += 1
        doc_id += 1
    elapsed_time = time.time() - start_time
    print(f'Feature extraction took {elapsed_time} seconds')
    print(features.shape)
    print(labels.shape)
    # id2label = dict([(id, label) for label, id in label2id.items()])
    dataset = np.column_stack((features, labels))
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
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
 
# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    start_time = time.time()
    tree = build_tree(train, max_depth, min_size)
    build_time = time.time() - start_time

    print_tree(tree)

    start_time = time.time()
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    predict_time = time.time() - start_time
    return predictions, build_time, predict_time


# The maximum size of the final vocabulary. It's a hyper-parameter. You can change it to        see what value gives the best performance.
MAX_VOCAB_SIZE = 10 #40000
N_FOLDS = 5 #10

MAX_DEPTH = 5
MIN_SIZE = 10

# Assuming this file is put under the same parent directoray as the data directory, and the     data directory is named "20news-train"
root_path = "./20news-train"

# Test CART on Bank Note dataset
seed(1)
# evaluate algorithm

scores, build_time, predict_time = evaluate_algorithm(root_path, decision_tree, MAX_DEPTH, MIN_SIZE)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print(build_time)
print(predict_time)