import os
import argparse
import h5py
import keras
import numpy as np
from imblearn.over_sampling import RandomOverSampler

np.random.seed(42)


def main(filename):
    dirname = os.path.dirname(__file__)
    in_path = os.path.join(dirname, 'data/interim/', filename)
    out_path = os.path.join(dirname, 'data/processed/', filename)

    with h5py.File(in_path, 'r') as dataset:
        x_train_original = np.array(dataset['train/X'])
        y_train_original = np.array(dataset['train/Y'])

    m = x_train_original.shape[0]
    num_classes = len(np.unique(y_train_original))

    resampler = RandomOverSampler()

    x_train_resampled, y_train_resampled = resampler.fit_resample(
        np.reshape(x_train_original,
                   (m, np.product(x_train_original.shape[1:]))),
        y_train_original
    )
    x_train_resampled = np.reshape(
        x_train_resampled,
        (x_train_resampled.shape[0], *x_train_original.shape[1:])
    )

    m = x_train_resampled.shape[0]
    idx = np.random.choice(m, int(m * 0.2))
    mask = np.ones(m, dtype=bool)
    mask[idx] = False

    x_train_split, x_val_split = x_train_resampled[mask], x_train_resampled[idx]
    y_train_split, y_val_split = y_train_resampled[mask], y_train_resampled[idx]

    x_train = x_train_split.astype('float16') / 255
    y_train = keras.utils.to_categorical(y_train_split, num_classes)

    x_val = x_val_split.astype('float16') / 255
    y_val = keras.utils.to_categorical(y_val_split, num_classes)

    with h5py.File(out_path, 'w') as file:
        file.create_dataset('val/X', data=x_val)
        file.create_dataset('val/Y', data=y_val)

        file.create_dataset('train/X', data=x_train)
        file.create_dataset('train/Y', data=y_train)

    print(f'Processed dataset saved at {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename',
        type=str,
        default='f3_32.h5',
        dest='filename',
        help='name of the interim data file'
    )

    args = parser.parse_args()

    main(args.filename)
