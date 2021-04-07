from sklearn import svm
from sklearn import metrics
import random
import numpy as np
import copy
from datetime import datetime


def info():
    print("""

    FILE: project/code/CuttleFish.py

    PROJECT TITLE: Face Recog. Using Gabor Filters with The CFA

    This file contains:

        Implementation of the CuttleFish Optimization Algorithm 
        
        reduce features by finding the most effective features
        (best features)
          

    Code by   :      Salar Adel Sabry
    Supervisor:  Mr. Haval Ismael Hussein
    """)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SolutionCellCFA:

    def __init__(self, fitness, selected_features, unselected_features):
        self.selected_features = copy.deepcopy(selected_features)
        self.unselected_features = copy.deepcopy(unselected_features)
        self.fitness = copy.deepcopy(fitness)


def sort_descending(population):
    # get fitness of each object and store in new list
    list_ = [sol.fitness for sol in population]

    # sort only the indices and store them in list_indices
    list_indices = np.argsort(list_)[::-1]

    # use list_indices (which holds the indices of all objects sorted descending order) to sort
    sorted_population = []
    for i in range(len(list_indices)):
        index_ = list_indices[i]
        sorted_population.append(population[index_])

    return sorted_population


# Evaluation Function
def evaluate(dataset, subset, classifier):
    x_train = copy.deepcopy(dataset[0])
    x_test = copy.deepcopy(dataset[1])
    y_train = copy.deepcopy(dataset[2])
    y_test = copy.deepcopy(dataset[3])

    classifier.fit(x_train[:, subset], y_train)
    y_pred = classifier.predict(x_test[:, subset])

    fitness = metrics.accuracy_score(y_test, y_pred)
    return fitness


def get_all_features_indices(sample):
    all_feat_indices = [i for i in range(len(sample))]
    return all_feat_indices


# CFA
def run_cfa(x_train, x_test, y_train, y_test, features_range, population_size, epoch, classifier=svm.SVC()):
    start_f, end_f = features_range
    # ------------------------------------------------------------------
    print(' - - - - - -  Begin CFA  - - - - - - ')
    print('Subset Size         : ', start_f, '-', end_f)
    print('All Features Size   : ', len(x_train[0]))
    print('Population Size (N) : ', population_size)
    print('Iterations (epoch)  : ', epoch)
    print('\nStarting CFA...\n')
    start_time = datetime.now()
    # ------------------------------------------------------------------

    # put all data and labels in list for ease when calling evaluation function
    dataset = [x_train, x_test, y_train, y_test]

    localBest = SolutionCellCFA(0, [], [])  # Best
    globalBest = SolutionCellCFA(0, [], [])  # AVBest

    # List of all features indices
    all_features = get_all_features_indices(x_train[0])

    ''' Initialize & Evaluation of Population | Steps 1,2 and 3 from CFA PseudoCode'''
    P = []

    # 1) Initialize the population p[N] with random subsets, where N represents the population size.
    for i in range(population_size):
        # Selected features
        s_feat = random.sample(all_features, k=random.randint(start_f, end_f))

        # UnSelected features
        uns_feat = list(set(all_features) - set(s_feat))

        # 2) Evaluate fitness of each subset (solution) using SVM classifier.
        fitness = evaluate(dataset, s_feat, classifier)

        # create cell/solution/ Member of P
        solution = SolutionCellCFA(fitness, s_feat, uns_feat)

        # 3) Keep the best solution in (Global Best Subset) and (Local Best Subset).
        if solution.fitness > localBest.fitness:
            localBest = copy.deepcopy(solution)
            globalBest = copy.deepcopy(solution)

        P.append(solution)

    ''' Delete 10% of Features | Step 4 of CFA PseudoCode'''
    # 4) randomly delete 10% of the features from selected features of best subset
    length_ = len(localBest.selected_features)
    list_ = localBest.selected_features
    take_random_90perc = int(float(length_) - float(length_) * 0.1)

    localBest.selected_features = random.sample(list_, take_random_90perc)
    localBest.unselected_features = list(set(all_features) - set(localBest.selected_features))

    ''' While | Step 5 of CFA Pseudo Code'''
    init_epoch = epoch
    while epoch > 0:
        if (init_epoch - epoch) % 10 == 0 and (init_epoch - epoch) != 0:
            print('RUNNING CFA | Number of Iterations Completed:', init_epoch - epoch)
        ''' Case 1 & 2 '''
        # Sort the population in descending order according to the fitness values
        P = sort_descending(P)

        # random.randint(a,b): returns an int between a(included) and b(included)
        # random.randrange(a,b): returns a number between a (included) and b (not included)
        k = random.randint(0, int(population_size / 2))

        for i in range(k):
            R = random.randrange(0, len(P[i].selected_features))
            V = len(P[i].selected_features) - R

            # Randomly choose R features from P[i].SelectedFeatures | Reflected Features
            Reflection = random.sample(P[i].selected_features, R)
            # Randomly choose V features from P[i].UnSelectedFeatures | Visible Features
            Visibility = random.sample(P[i].unselected_features, V)

            newSubset = Reflection + Visibility

            # Evaluate newSubset using Classifier
            fitness = evaluate(dataset, newSubset, classifier)

            if fitness > P[i].fitness:
                unSelFeatures = list(set(all_features) - set(newSubset))
                P[i] = SolutionCellCFA(fitness, newSubset, unSelFeatures)

            if fitness > globalBest.fitness:
                unSelFeatures = list(set(all_features) - set(newSubset))
                globalBest = SolutionCellCFA(fitness, newSubset, unSelFeatures)

        ''' Case 3 & 4 '''
        t = 10
        newSubset = copy.deepcopy(localBest)
        bss = len(newSubset.selected_features)
        bus = len(newSubset.unselected_features)
        bsf = range(bss)
        buf = range(bus)

        for i in range(t):
            # randomly choose 10% of features indices from newSubset.SelectedFeatures
            R = random.sample(bsf, int(float(bss) * 0.1))
            # randomly choose 10% of features indices from newSubset.UnSelectedFeatures
            V = random.sample(buf, int(float(bss) * 0.1))

            for index1, index2 in zip(R, V):
                temp = newSubset.selected_features[index1]
                newSubset.selected_features[index1] = newSubset.unselected_features[index2]
                newSubset.unselected_features[index2] = temp

            fitness = evaluate(dataset, newSubset.selected_features, classifier)
            if fitness > localBest.fitness:
                localBest = copy.deepcopy(newSubset)
                localBest.fitness = fitness

        ''' Case 5 '''
        m = 10
        length_ = len(globalBest.selected_features)
        list_features = globalBest.selected_features

        for i in range(m):
            newSubset = random.sample(list_features, int(float(length_) - float(length_ * 0.1)))
            fitness = evaluate(dataset, newSubset, classifier)

            if fitness > localBest.fitness:
                localBest.selected_features = copy.deepcopy(newSubset)
                localBest.unselected_features = list(set(all_features) - set(newSubset))
                localBest.fitness = copy.deepcopy(fitness)

        ''' Case 6 '''
        for i in range(k, population_size):
            # randomly generate newSubset
            newSubset = random.sample(all_features, k=random.randint(start_f, end_f))
            # evaluate new subset
            fitness = evaluate(dataset, newSubset, classifier)

            if fitness > P[i].fitness:
                P[i].selected_features = copy.deepcopy(newSubset)
                P[i].unselected_features = list(set(all_features) - set(newSubset))
                P[i].fitness = copy.deepcopy(fitness)

            if fitness > globalBest.fitness:
                globalBest.selected_features = copy.deepcopy(newSubset)
                globalBest.fitness = copy.deepcopy(fitness)
        epoch = epoch - 1

    ''' Step 6 of Pseudo Code'''
    if localBest.fitness > globalBest.fitness:
        bestSolution = copy.deepcopy(localBest)
    else:
        bestSolution = copy.deepcopy(globalBest)

    # ------------------------------------------------------------------
    print('\n------------------------------------------------------------------')
    print('CFA | Time Elapsed:', datetime.now() - start_time)
    print('CFA | Accuracy:', bestSolution.fitness)
    print('CFA | Number of Features in Best Subset:', len(bestSolution.selected_features))
    print('CFA | Best Subset:', bestSolution.selected_features)
    print('------------------------------------------------------------------\n')
    print(' - - - - - -   End CFA   - - - - - - ')
    # ------------------------------------------------------------------

    return bestSolution.selected_features, bestSolution.fitness


# - - - - - - - - - - - - - - - - - - - - - - - - - - -'
''' - - - - - - - - - - - TEST - - - - - - - - - - - '''
# - - - - - - - - - - - - - - - - - - - - - - - - - - -'
import pandas as pd
from sklearn.model_selection import train_test_split
# https://realpython.com/train-test-split-python-data/


def test():
    # https://machinelearningmastery.com/standard-machine-learning-datasets/
    dataset_dict = {'Sonar': [r'CFA Test/sonar.all-data', 10],  # Binary Class.: 0.0=M | 1.0=R
                    'Ionosphere': [r'CFA Test/ionosphere.data', 8],  # Binary Class.: 0.0=B | 1.0=G
                    }
    for key, value in dataset_dict.items():
        dataset_ = read_data_from_csv(value[0], header=None)
        x, y = split_data_labels(dataset_)
        x_train_, x_test_, y_train_, y_test_ = train_test_split(np.array(x), np.array(y), train_size=0.25)

        dataset_ = [x_train_, x_test_, y_train_, y_test_]

        all_features_indices = get_all_features_indices(x_test_[0])

        n_epoch = 100
        num_of_features = (5, 10)
        population_size = 20
        bestFeatures_indices = run_cfa(x_train_, x_test_, y_train_, y_test_, num_of_features, population_size, n_epoch)

        fitness_noCFA = evaluate(dataset_, all_features_indices, classifier=svm.SVC())
        print('accuracy using all features:', fitness_noCFA)

        fitness_CFA = evaluate(dataset_, bestFeatures_indices, classifier=svm.SVC())
        print('accuracy using CFA features subset:', fitness_CFA)


def split_data_labels(data_list):
    # split dataset to data and labels
    data, label = [], []
    for row in data_list:
        data.append(row[:-1])
        label.append(row[-1])

    return data, label


def read_data_from_csv(path, header=None):
    dataset_ = pd.read_csv(path, header=header)

    # convert to list
    data_list = [
        [row[cell] for cell in range(len(row)) if not pd.isnull(row[cell])]
        for row in dataset_.values[:]]

    return data_list


if __name__ == '__main__':
    test()
    info()
