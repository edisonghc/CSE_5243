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
from math import sqrt


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


# Get list of files
def load_files(root_path):
    print(f'|   Reading files from {root_path}')
    start_time = time.time()

    # Only keep the data dictionaries and ignore possible system files like .DS_Store
    folders = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    i = 0
    list_files = list()
    for folder in folders:
        print(f'|   |   {i+1}) Reading data from {folder}')
        files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
        i += 1
        for file in files:
            list_files.append(file)

    elapsed_time = time.time() - start_time
    print(f'|   Reading files took {elapsed_time:.2f} seconds')
    print()
    return list_files


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
    # count_cv = 0
    for i in range(len(folds)):

        # A switch to run the algorithm just once for efficiency
        if i > 0:
            continue

        print(f'|   {i+1}th iteration')
        train_set = list()
        for j in range(len(folds)):
            if not i == j:
                train_set = np.append(train_set, copy.deepcopy(folds[j]))

        print(f'|   |   Building feature matrix on training data')
        train_vocab = construct_vocab(train_set)
        # count_cv += 1
        train_data, label2id = extract_feature(train_set, train_vocab)
        minmax = dataset_minmax(train_data)
        normalize_dataset(train_data,minmax)
        print()

        test_set = list(folds[i])
        print(f'|   |   Building feature matrix on testing data')
        test_data_with_label, _ = extract_feature(test_set, train_vocab, label2id)
        normalize_dataset(test_data_with_label,minmax)
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
def construct_vocab(files,count_cv=0):
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
        folder_name = [os.path.basename(folder) for folder in folders]
        label2id = dict([(label, id) for id, label in enumerate(folder_name)])
    for file in files:
        folder = os.path.dirname(file)
        label = os.path.basename(folder)
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


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
 
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    start_time = time.time()
    print(f'|   |   |   Running KNN')
    predictions = list()
    i = 0
    for row in test:
        print(f'|   |   |   |   Predicting class for {i}th row')
        predic_start_time = time.time()
        output = predict_classification(train, row, num_neighbors)
        predic_time = time.time() - predic_start_time
        print(f'|   |   |   |   |   Predicting class took {predic_time:.2f} seconds')
        predictions.append(output)
        i += 1
    run_time = time.time() - start_time
    print(f'|   |   |   Running KNN took {run_time/60:.2f} minutes')
    return(predictions)
 

run_start_time = time.time()
# The maximum size of the final vocabulary. It's a hyper-parameter. You can change it to see what value gives the best performance.
MAX_VOCAB_SIZE = 40000
N_FOLDS = 5 #10
NUM_NEIGHBORS = 20

# Assuming this file is put under the same parent directoray as the data directory, and the data directory is named "20news-train"
train_path = "./20news-train"
test_path = "./20news-test"

# Test CART
seed(1)
# evaluate algorithm

scores = evaluate_algorithm(train_path, k_nearest_neighbors, NUM_NEIGHBORS)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

print("Scores:", *(f"{s:.3f}%" for s in scores))
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
print()
run_time = time.time() - run_start_time
print(f'The program ran for {run_time/60:.2f} minutes')
print('Done!')



# start_time = time.time()

# train_set = load_files(train_path)
# test_set = load_files(test_path)

# print(f'|   |   Building feature matrix on training data from: {train_path}')
# train_vocab = construct_vocab(train_set)
# train_data, label2id = extract_feature(train_set, train_vocab)

# print(f'|   |   Building feature matrix on testing data from: {test_path}')
# test_data_with_label, _ = extract_feature(test_set, train_vocab, label2id)
# test_data_no_label = test_data_with_label[:,:-1]

# print(f'|   |   Running decision_tree')
# predicted, tree = decision_tree(train_data, test_data_no_label, MAX_DEPTH, MIN_SIZE)

# print(f'|   |   Caculating the accuracy')
# actual = [row[-1] for row in test_data_with_label]
# accuracy = accuracy_metric(actual, predicted)
# print(f'|   |   Accuracy is {accuracy:.2f}%')

# print_tree(tree)

# elapsed_time = time.time() - start_time
# print(f'Evaluating decision_tree took {elapsed_time/60:.2f} minutes')
