import cv2
import glob2 as glob


def info():
    print("""
    
    FILE: project/code/Read_IMG.py
    
    PROJECT TITLE: Face Recog. Using Gabor Filters with The CFA
    
    This file contains the functions used to read a dataset that is stored in the form of
    image files in a directory of subdirectories
    
    
    
    Code by   :      Salar Adel Sabry
    Supervisor:  Mr. Haval Ismael Hussein
    """)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def read_images(path, ext='.jpg'):
    path = r'' + path + '**'
    x, y = [], []

    for img_path in glob.glob(path + r'\*'+ext, recursive=True):
        # read image
        img = cv2.imread(img_path)

        # grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize
        resized_img = cv2.resize(gray, (65, 65))

        x.append(resized_img)
        y.append(img_path.split('\\')[-2])
        # print('reading files... (' + str(img_path) + ')')

    print('read ', len(y), ' files success.')
    return x, y


if __name__ == '__main__':
    info()
