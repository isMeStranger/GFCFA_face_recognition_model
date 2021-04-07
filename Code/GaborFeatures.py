import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def info():
    print("""

    FILE: project/code/GaborFeatures.py

    PROJECT TITLE: Face Recog. Using Gabor Filters with The CFA

    This file contains:
    
        Class Gabor:
            private var __FiltersArray
            
            def generate filters
            def extract features
            def get_size (of array of filters) 


    Code by   :      Salar Adel Sabry
    Supervisor:  Mr. Haval Ismael Hussein
    """)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Gabor:
    def __init__(self):
        self.__FiltersArray = []

    def generate_filters(self, u, v, m, n):
        print('Generating Gabor Banks...')
        fmax = 0.25
        gama = np.sqrt(2)
        eta = np.sqrt(2)

        self.__FiltersArray = []

        for i in range(1, u + 1):
            fu = fmax / ((np.sqrt(2)) ** (i - 1))
            alpha = fu / gama
            beta = fu / eta

            for j in range(1, v + 1):
                tetav = ((j - 1) / v) * np.pi
                gFilter = np.zeros((m, n), np.complex128)

                for x in range(1, m + 1):
                    for y in range(1, n + 1):
                        xprime = (x - ((m + 1) / 2)) * np.cos(tetav) + (y - ((n + 1) / 2)) * np.sin(tetav)
                        yprime = -(x - ((m + 1) / 2)) * np.sin(tetav) + (y - ((n + 1) / 2)) * np.cos(tetav)
                        gFilter[x - 1, y - 1] = ((fu ** 2) / (np.pi * gama * eta)) * np.exp(
                            -((alpha ** 2) * (xprime ** 2) + (beta ** 2) * (yprime ** 2))) * np.exp(
                            1j * 2 * np.pi * fu * xprime)
                self.__FiltersArray.append(gFilter)

        print('Gabor Banks Generation Complete.\n')

    def extract_features(self, img, d1, d2):

        try:
            img = np.double(img)
        except Exception as ex:
            print('Image might not be in grayscale (contains 3 channels):', ex)

        featureVector = []
        dict = { }
        uk = 1
        for i in self.__FiltersArray:
            result1 = signal.fftconvolve(img, i.real)
            dict[str(uk)] = result1
            uk+=1
            result2 = signal.fftconvolve(img, i.imag)
            field_total = result1 + (result2 * 1j)
            gaborAbs = abs(field_total)
            gaborAbs = gaborAbs[::d1, ::d2].ravel()
            featureVector.append(gaborAbs)

        display_multiple_img(dict,rows=5,cols=8)
        return np.array(featureVector).flatten()

    def get_size(self):
        num_rows = len(self.__FiltersArray)
        num_cols = len(self.__FiltersArray[0])
        print("""
        2D Array(DataTypes: List | List of Lists)
            number of rows: {0}
            number of cols: {1}
        """, num_rows, num_cols)

        return num_rows, num_cols


# test

import cv2


def display_multiple_img(images, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title],cmap='gray')
        ax.ravel()[ind].set_title(' ')
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()


def test():
    gabor = Gabor()
    IMAGE = cv2.imread(r'GaborTest/test.pgm')
    gray = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)

    gabor.generate_filters(5, 8, 39, 39)
    rows, cols = gabor.get_size()
    print(rows, cols)

    # down sampling 1
    d1 = 4
    # down sampling 2
    d2 = 4

    feature_vector = gabor.extract_features(gray, d1, d2)

    print(len(feature_vector))


if __name__ == '__main__':
    info()
    test()
