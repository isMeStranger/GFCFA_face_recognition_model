from Code.Read_IMG import read_images
import Code.SaveMyPickles as MyPkls
import Code.Message as Msgs
import Code.MyFRClassifiers as Classifier
import Code.CuttleFish as CFA

from Code.PlotTable import plot_table
from Code.GaborFeatures import Gabor

from sklearn.neighbors import KNeighborsClassifier as KNN_Clf

import statistics
import matplotlib.pyplot as plt

''' Stage 1 | Feature Extraction & Data Preparation'''

path = r'Datasets/ATT/'
ext = '.pgm'
number_of_images_per_person = 10

# Create Pickles Dir if it doesn't exist
Msgs.create_dir('Pickles')

if Msgs.yes_no_msg('Read dataset from ' + path + '?'):
    # Read Images
    x, y = read_images(path, ext)

    # Generate gabor filters
    gabor = Gabor()
    gabor.generate_filters(5, 8, 39, 39)

    # Extract feature vectors
    feature_vectors = []
    print('Now extracting features using Gabor banks...')
    for img in x:
        # down-sampling
        d1, d2 = 4, 4

        # Extract single feature vector
        _vector = gabor.extract_features(img, d1, d2)
        feature_vectors.append(_vector)

    print('len of each feature vector is:', len(feature_vectors[0]), 'features.')
    print('Gabor feature extraction complete.')

    # Gabor object is not needed anymore
    del gabor

    # convert labels from STRINGS to INT numbers
    print('\nConverting labels from text to numbers...')
    for label, i in zip(y, range(len(y))):
        y[i] = int(label[1:])

    print('Conversion complete.')

    # Export Features & Labels
    if Msgs.yes_no_msg('Save Gabor Features as Pickle?'):
        MyPkls.save_pickle(r'Pickles/feature_vectors.pkl', feature_vectors)
        MyPkls.save_pickle(r'Pickles/labels.pkl', y)

# path is not needed anymore
del path

''' Stage 2 | Classification using All Features (No CFA) '''

# if we skipped feature extraction, we can load existing features
if Msgs.yes_no_msg("Load pickle file?\n"
                   "NOTE: if you haven't extracted Gabor Features, then you must load a file.\n\t"):
    feature_vectors = MyPkls.load_pickle(r'Pickles/feature_vectors.pkl')
    y = MyPkls.load_pickle(r'Pickles/labels.pkl')

# Support Vector Machine
if Msgs.yes_no_msg('Run SVM? (No CFA)'):
    # calculate the average of 10 iterations of using i images as training and the rest as testing
    average = []

    for i in range(1, number_of_images_per_person):
        acc_list = []
        macro_list = []
        weighted_list = []
        for j in range(10):
            number_of_training_images = i
            x_train, x_test, y_train, y_test = Classifier.prepare_data(feature_vectors, y,
                                                                       number_of_images_per_person,
                                                                       number_of_training_images)

            acc, macroAV, weightedAV = Classifier.run_svm(x_train, x_test, y_train, y_test,
                                                          iterations=i, jiterations=j)[0:3]
            acc_list.append(acc)
            macro_list.append(macroAV)
            weighted_list.append(weightedAV)

        # average of 10 iterations of using i images as training and the rest for testing
        average.append([str(i),  # Train Images Used
                        statistics.mean(acc_list),  # Mean Acc
                        statistics.mean(macro_list),  # Mean Macro
                        statistics.mean(weighted_list),  # Mean Weighted
                        str(number_of_images_per_person - i)  # Test Images Used
                        ])

    Msgs.create_dir('Results')
    MyPkls.save_pickle('Results/average_svm_noCFA.pkl', average)

    plot_table(title='Support Vector Machine: Using All Features (No CFA)', data=average)
    plt.savefig('Results/svm_av_table_noCFA.png')
    # plt.show()
    plt.close

# K Nearest Neighbors
if Msgs.yes_no_msg('Run KNN? (No CFA)'):
    # calculate the average of 10 iterations of using i images as training and the rest as testing
    average = []

    for i in range(1, number_of_images_per_person):
        acc_list = []
        macro_list = []
        weighted_list = []
        for j in range(10):
            number_of_training_images = i
            x_train, x_test, y_train, y_test = Classifier.prepare_data(feature_vectors, y,
                                                                       number_of_images_per_person,
                                                                       number_of_training_images)

            acc, macroAV, weightedAV = Classifier.run_knn(x_train, x_test, y_train, y_test,
                                                          iterations=i, jiterations=j, n_neighbors=5)[0:3]
            acc_list.append(acc)
            macro_list.append(macroAV)
            weighted_list.append(weightedAV)

        # average of 10 iterations of using i images as training and the rest for testing
        average.append([str(i),  # Train Images Used
                        statistics.mean(acc_list),  # Mean Acc
                        statistics.mean(macro_list),  # Mean Macro
                        statistics.mean(weighted_list),  # Mean Weighted
                        str(number_of_images_per_person - i)  # Test Images Used
                        ])

    Msgs.create_dir('Results')
    MyPkls.save_pickle('Results/average_knn_noCFA.pkl', average)

    plot_table(title='K Nearest Neighbors: Using All Features (No CFA)', data=average)
    plt.savefig('Results/knn_av_table_noCFA.png')
    # plt.show()
    plt.close

''' Stage 3 | Classification using CFA Features Subset '''

if Msgs.yes_no_msg('Run CFA?'):
    # calculate the average of 10 iterations of using i images as training and the rest as testing
    average_svm = []
    average_knn = []

    svm_dict = {}
    knn_dict = {}

    for i in range(1, number_of_images_per_person):
        acc_list_svm = []
        macro_list_svm = []
        weighted_list_svm = []

        acc_list_knn = []
        macro_list_knn = []
        weighted_list_knn = []

        for j in range(10):
            x_train, x_test, y_train, y_test = Classifier.prepare_data(feature_vectors, y, number_of_images_per_person, i)
            SVM_bestFeatures_CFA, fitness1= CFA.run_cfa(x_train, x_test, y_train, y_test, (500, 2000), 25, 80)
            KNN_bestFeatures_CFA, fitness2 = CFA.run_cfa(x_train, x_test, y_train, y_test, (500, 2000), 25, 80, KNN_Clf(5))

            acc1, macroAV1, weightedAV1, clsreport1, cm1 = Classifier.run_svm(x_train[:, SVM_bestFeatures_CFA],
                                                             x_test[:, SVM_bestFeatures_CFA],
                                                             y_train,
                                                             y_test,
                                                             iterations=i, jiterations=j,  cfa=True)
            acc2, macroAV2, weightedAV2, clsreport2, cm2 = Classifier.run_knn(x_train[:, KNN_bestFeatures_CFA],
                                                             x_test[:, KNN_bestFeatures_CFA],
                                                             y_train,
                                                             y_test,
                                                             iterations=i, jiterations=j,
                                                             n_neighbors=5, cfa=True)
            acc_list_svm.append(acc1)
            macro_list_svm.append(macroAV1)
            weighted_list_svm.append(weightedAV1)

            acc_list_knn.append(acc2)
            macro_list_knn.append(macroAV2)
            weighted_list_knn.append(weightedAV2)

            svm_dict['svm iteration ' + str(i) + str(j)] = \
                [SVM_bestFeatures_CFA, acc1, macroAV1, weightedAV1, clsreport1, cm1]

            knn_dict['knn iteration ' + str(i) + str(j)] = \
                [KNN_bestFeatures_CFA, acc2, macroAV2, weightedAV2, clsreport2, cm2]

        average_svm.append([str(i),  # Train Images Used
                            statistics.mean(acc_list_svm),  # Mean Acc
                            statistics.mean(macro_list_svm),  # Mean Macro
                            statistics.mean(weighted_list_svm),  # Mean Weighted
                            str(number_of_images_per_person - i)  # Test Images Used
                            ])

        average_knn.append([str(i),  # Train Images Used
                            statistics.mean(acc_list_knn),  # Mean Acc
                            statistics.mean(macro_list_knn),  # Mean Macro
                            statistics.mean(weighted_list_knn),  # Mean Weighted
                            str(number_of_images_per_person - i)  # Test Images Used
                            ])

    # save pickles
    Msgs.create_dir('Results')
    MyPkls.save_pickle('Results/dict_svm_CFA.pkl', svm_dict)
    MyPkls.save_pickle('Results/dict_knn_CFA.pkl', knn_dict)

    MyPkls.save_pickle('Results/average_svm_CFA.pkl', average_svm)
    MyPkls.save_pickle('Results/average_knn_CFA.pkl', average_knn)

    # plot and save tables as png
    plot_table(title='Support Vector Machine: Using CFA Features Subsets', data=average_svm)
    plt.savefig('Results/svm_av_table_CFA.png')
    # plt.show()
    plt.close()

    plot_table(title='K Nearest Neighbors: Using CFA Features Subsets', data=average_knn)
    plt.savefig('Results/knn_av_table_CFA.png')
    # plt.show()
    plt.close()
