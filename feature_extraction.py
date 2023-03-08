import cv2 as cv
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize

from sklearn.datasets import dump_svmlight_file


def getAverageDiagonalOnZone(img, i, j):
    diagonals = []

    for k in range(0, 19):
        numDiag = 0

        l = min(9, k)
        c = max(0, k - 10)
        while l >= 0 and c >= 0:
            if img[i + l][j + c] == 1:
                numDiag += 1

            l -= 1
            c -= 1

        diagonals.append(numDiag)

    return np.mean(diagonals)


def feature_extraction(img):
    features = []

    for i in range(0, 90, 10):
        for j in range(0, 60, 10):
            features.append(getAverageDiagonalOnZone(img, i, j))

    return features


def preprocess_image(file):
    img = cv.imread(file)
    img = rgb2gray(img)

    img = resize(img, (90, 60), anti_aliasing=True)

    t = threshold_mean(img)
    thresh_image = img > t

    return thresh_image


def main():
    samples_features = []
    labels = []

    lines = []
    with open('digits/files.txt', 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        file, label = lines[i].split(' ')
        file = f'digits/{file}'

        img = preprocess_image(file)
        features = feature_extraction(img)

        samples_features.append(features)
        labels.append(int(label))

    dump_svmlight_file(samples_features, labels, 'features.txt')


if __name__ == '__main__':
    main()
