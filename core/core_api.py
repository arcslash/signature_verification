import core.model
from sklearn.model_selection import train_test_split
# Maximum signature size
import os
import numpy as np
from PIL import Image
canvas_size = (952, 1360)


def load_dataset(path='data/dataset/'):
    #loading dataset to X
    X = {}
    Y = {}
    original_img_path = os.path.join(path, 'genuine/')
    forged_img_path = os.path.join(path, 'forged/')
    original_imgs = os.listdir(original_img_path)
    forged_imgs = os.listdir(forged_img_path)

    #load data into memory
    current_iter = 1
    for original in original_imgs:
        print(original)
        image = Image.open(os.path.join(original_img_path, original))
        image = np.asarray(image)
        print("Image shape:", image.shape)
        if not original.split("_")[1] in X.keys():
            Y[original.split("_")[1]] = []
            X[original.split("_")[1]] = []
            X[original.split("_")[1]].append(image)
            Y[original.split("_")[1]].append(1)
        else:
            X[original.split("_")[1]].append(image)
            Y[original.split("_")[1]].append(1)

    for forge in forged_imgs:
        print(forge)
        image = Image.open(os.path.join(forged_img_path, forge))
        image = np.asarray(image)
        print("Image shape:", image.shape)
        if not original.split("_")[1] in X.keys():
            Y[forge.split("_")[1]] = []
            X[forge.split("_")[1]] = []
            X[forge.split("_")[1]].append(image)
            Y[forge.split("_")[1]].append(0)
        else:
            X[forge.split("_")[1]].append(image)
            Y[forge.split("_")[1]].append(0)




    X = list(X.values())
    X = np.array(X)
    print("shape:", X.shape)
    Y = list(Y.values())
    Y = np.array(Y)
    print("shape:", Y.shape)

    return X, Y








if __name__ == '__main__':
    load_dataset("../data/dataset/")



