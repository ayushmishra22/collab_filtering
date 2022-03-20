import numpy as np
import sys
import os
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

PATH = os.path.join(os.getcwd(), "netflix")
KNN = 18

def read_dataset(filepath, filename):
    abs_path = os.path.join(filepath, filename)
    file = open(abs_path, 'r')
    data = file.readlines()
    user_ids = []
    movie_ids = []
    train_movie_user = {}
    for i in data:
        l = i.strip().split(',')
        l = [int(float(j)) for j in l]
        user_ids.append(l[1])
        movie_ids.append(l[0])
        if l[0] in train_movie_user:
            train_movie_user[l[0]].append(l[1])
        else:
            train_movie_user[l[0]] = [l[1]]
    unique_user_ids = sorted(set(user_ids))
    unique_movie_ids = sorted(set(movie_ids))
    user_ids_map = {}
    movie_ids_map = {}
    for a,b in enumerate(unique_user_ids):
        user_ids_map[b] = a
    for a,b in enumerate(unique_movie_ids):
        movie_ids_map[b] = a
    matrix = np.zeros((len(unique_user_ids), len(unique_movie_ids)))
    for i in data:
        l = i.strip().split(',')
        l = [int(float(j)) for j in l]
        matrix[user_ids_map[l[1]], movie_ids_map[l[0]]] = l[2]
    return matrix, user_ids_map, movie_ids_map, train_movie_user

def read_test_dataset(filepath, filename, user_ids_map, movie_ids_map):
    abs_path = os.path.join(filepath, filename)
    file = open(abs_path, 'r')
    data = file.readlines()
    user_ids = []
    movie_ids = []
    test_list = []
    ratings = []
    for i in data:
        l = i.strip().split(',')
        l = [int(float(j)) for j in l]
        user_ids.append(l[1])
        movie_ids.append(l[0])
        tl = []
        tl.append(l[0])
        tl.append(l[1])
        tl.append(l[2])
        ratings.append(l[2])
        test_list.append(tl)
    test_matrix = np.zeros((len(user_ids_map), len(movie_ids_map)))
    return test_matrix, test_list, ratings

def compute_mean(matrix):
    rows, columns = matrix.shape
    m = np.sum(matrix, axis=1)
    non_zero = np.count_nonzero(matrix, axis=1)
    mean = m/non_zero
    return mean

def compute_weights(matrix, k_user_list, user_ids_map, test_user, mean):
    no_of_users = len(k_user_list)
    test_user_enum = user_ids_map[test_user]
    weight_matrix = {}
    for i in k_user_list:
        row = user_ids_map[i]
        weight_matrix[i] = np.dot(matrix[row], matrix[test_user_enum])/(np.linalg.norm(matrix[row]) * np.linalg.norm(matrix[test_user_enum]))
        if math.isnan(weight_matrix[i]):
            weight_matrix[i] = 0
    return weight_matrix

def k_nearest_neighbor(matrix, user_list, user_ids_map, k, test_user_enum, dist_mat):
    no_of_users = len(matrix)
    distance = {}
    for i in user_list:
        if dist_mat[user_ids_map[i]][test_user_enum] == 0:
            row = user_ids_map[i]
            dist_mat[row][test_user_enum] = np.linalg.norm(matrix[test_user_enum] - matrix[user_ids_map[i]])
            distance[i] = dist_mat[user_ids_map[i]][test_user_enum]
        else:
            distance[i] = dist_mat[user_ids_map[i]][test_user_enum]
    dist = sorted(distance.items(), key=lambda x:x[1])
    indices = []
    for i in range(len(dist)):
        indices.append(dist[i][0])
    k_user_indices = indices[:k]
    return k_user_indices

def rms_error(true, predicted):
    error = mean_squared_error(true, predicted, squared=False)
    print("Root Mean Square Error = {}".format(error))

def mean_abs_error(true, predicted):
    error = mean_absolute_error(true, predicted)
    print("Mean Absolute Error = {}".format(error))



def predict_collaborative_filtering(train_matrix, movie_user, test_list, test_matrix, user_ids_map, movie_ids_map, distance, k, mean):
    m, n = train_matrix.shape
    centered_train_matrix = np.zeros((m, n))
    for i in range(m):
        for j in np.where(train_matrix[i] != 0):
            centered_train_matrix[i, j] = train_matrix[i, j] - mean[i]
    prediction = []
    for g in range(len(test_list)):
        print("Test row = {}".format(g))
        test_movie = test_list[g][0]
        test_user = test_list[g][1]
        user_list = movie_user[test_movie]
        max_users = int(len(user_list)*0.1)
        k_indices = k_nearest_neighbor(train_matrix, user_list[:max_users], user_ids_map, k, user_ids_map[test_user], distance)
        weights = compute_weights(centered_train_matrix, k_indices, user_ids_map, test_user, mean)

        s = 0
        kappa = 0
        for i in k_indices:
            s = s + abs(weights[i])
        if s != 0:
            kappa = 1 / s
        sum = 0
        for i in k_indices:
            sum = sum + (weights[i] * centered_train_matrix[user_ids_map[i], movie_ids_map[test_movie]])
        p = mean[user_ids_map[test_user]] + (kappa * sum)
        prediction.append(p)
    return prediction


if __name__ == '__main__':
    #read the dataset, return a numpy data matrix
    train_matrix, user_ids_map, movie_ids_map, train_movie_user = read_dataset(PATH, "TrainingRatings.txt")
    test_matrix, test_list, ratings = read_test_dataset(PATH, "TestingRatings.txt", user_ids_map, movie_ids_map)
    m, n = train_matrix.shape
    distances = np.zeros((m,m))
    mean = compute_mean(train_matrix)
    predicted = predict_collaborative_filtering(train_matrix, train_movie_user, test_list, test_matrix, user_ids_map, movie_ids_map, distances, KNN, mean)
    rms_error(ratings, predicted)
    mean_abs_error(ratings, predicted)
