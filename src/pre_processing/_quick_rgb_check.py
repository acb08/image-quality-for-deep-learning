import numpy as np
from src.utils.functions import load_data_vectors


if __name__ == '__main__':

    directory = r'/home/acb6595/places/datasets/train/0003trn-np-rgb-checkout/train_split'
    file_names = ['train_vectors_0.npz',
                  'train_vectors_1.npz',
                  'train_vectors_2.npz',
                  'train_vectors_3.npz',
                  'train_vectors_4.npz',
                  'train_vectors_5.npz',
                  'train_vectors_6.npz',
                  'train_vectors_0.npz']

    for file_name in file_names:
        image_array, label_array = load_data_vectors(file_name, directory)

        for i in range(len(label_array)):
            image = image_array[i]
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            print('label:', label_array[i])
            print('rgb means:', np.mean(r), np.mean(g), np.mean(b))
            print('rgb stds: ', np.std(r), np.std(g), np.std(b))
            print(np.mean(r) == np.mean(g) == np.mean(b))
            print('\n')



