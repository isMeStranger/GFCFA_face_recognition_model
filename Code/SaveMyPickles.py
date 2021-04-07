import pickle


def info():
    print("""

    FILE: project/code/SaveMyPickles.py

    PROJECT TITLE: Face Recog. Using Gabor Filters with The CFA

    This file contains the functions used to:
    
        * Load saved object data from an existing pickle
        * Save object data into a new or overwrite existing pickle


    Code by   :      Salar Adel Sabry
    Supervisor:  Mr. Haval Ismael Hussein
    """)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def save_pickle(pickle_path, obj, mode='wb'):

    with open(pickle_path, mode) as file:
        pickle.dump(obj, file, -1)
        print('save pickle success.(' + pickle_path + ')')


def load_pickle(pickle_path, mode='rb'):

    with open(pickle_path, mode) as file:
        obj = pickle.load(file)
        print('load pickle success. (' + pickle_path + ')')

    return obj


''' Test '''


def test():
    print('\tTEST:')
    my_list = [[234, 2353, 235], [234, 2353, 235], [234, 2353, 235]]
    save_pickle(r'test.pkl', my_list)
    read_list = load_pickle(r'test.pkl')
    print(read_list)


if __name__ == '__main__':
    # test()
    info()
