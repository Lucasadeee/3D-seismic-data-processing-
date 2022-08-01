import os
import argparse
import numpy as np
import h5py


def save_dataset(path, X_train, Y_train):
    f = h5py.File(path, 'w')
    f.create_dataset('train/X', data=X_train)
    f.create_dataset('train/Y', data=Y_train)


def sample_well_locations(n_wells, x_range, y_range):
    well_locations = []
    x_coords = np.random.choice(x_range, n_wells)
    y_coords = np.random.choice(y_range, n_wells)

    for i in range(n_wells):
        well_locations.append((x_coords[i], y_coords[i]))
    return well_locations


def main(seed=42, image_size=8, out_filename='f3_32.h5'):
    print('Creating dataset...\n')

    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, 'data/raw/')
    dataset_path = os.path.join(
        dirname, '../interim', out_filename)

    seismic_cube = np.load(os.path.join(dirname, 'train_seismic.npy'))
    facies_cube = np.load(os.path.join(dirname, 'train_labels.npy'))

    width, depth, height = seismic_cube.shape

    x_range = width - image_size + 1
    y_range = depth - image_size + 1
    z_range = height - image_size + 1

    np.random.seed(seed)

    # inline, xline of F3 wells inside training data, corrected to np index
    well_locations = np.asarray([
        (62, 36),
        (18, 160),
        (152, 155),
        (126, 91),
        (53, 130),
        (56, 129),
        (50, 600)
    ])
    n_wells = well_locations.shape[0]
    m_train = n_wells * z_range

    seismic_cube -= np.min(seismic_cube)
    seismic_cube /= np.max(seismic_cube)
    seismic_cube *= 255

    X_train = np.empty((m_train, image_size, image_size, 3), dtype='uint8')
    Y_train = np.empty((m_train, 1), dtype='int8')

    mid_idx = int(np.median(range(image_size)))

    j = 0
    for x, y in well_locations:
        x -= int(image_size/2) - 1
        y -= int(image_size/2) - 1
        print(f'x: {x}, y: {y}')
        for z in range(z_range):

            sample = seismic_cube[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size,
            ]

            image = np.moveaxis(
                np.array([
                    sample[mid_idx, :, :],  # Red    -> height
                    sample[:, mid_idx, :],  # Green  -> width
                    sample[:, :, mid_idx],  # Blue   -> depth
                ]),
                0,
                -1
            )

            facies = facies_cube[
                x:x+image_size,
                y:y+image_size,
                z:z+image_size,
            ][mid_idx, mid_idx, mid_idx]

            X_train[j] = image
            Y_train[j] = facies
            j += 1

    print(f'Seed: {seed}')
    print(f'Input shape: {width, depth, height} (x, y, z)')
    print(f'#Wells sampled: {n_wells}')
    print(f'Sampled well locations:\n{well_locations}\n')

    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')

    print(f'Saving dataset as "{dataset_path}"\n...')
    save_dataset(
        dataset_path,
        X_train, Y_train,
    )
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--image_size',
        type=int,
        default=8,
        dest='image_size',
        help='size of dataset images (image_size x image_size)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        dest='seed',
        help='RNG initial seed'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='f3_32.h5',
        dest='out_filename',
        help='Output file name'
    )
    args = parser.parse_args()

    main(
        seed=args.seed,
        image_size=args.image_size,
        out_filename=args.out_filename
    )
