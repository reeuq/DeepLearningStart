from __future__ import print_function
from urllib.request import urlretrieve
import numpy as np
import os
import sys
import tarfile
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

url = 'http://yaroslavvb.com/upload/notMNIST/'
last_percent_reported = None
data_root = 'F:\\'  # Change me to store data elsewhere

# def download_progress_hook(count, blockSize, totalSize):
#     """A hook to report the progress of a download. This is mostly intended for users with
#     slow internet connections. Reports every 5% change in download progress.
#     """
#     global last_percent_reported
#     percent = int(count * blockSize * 100 / totalSize)
#
#     if last_percent_reported != percent:
#         if percent % 5 == 0:
#             sys.stdout.write("%s%%" % percent)
#             sys.stdout.flush()
#         else:
#             sys.stdout.write(".")
#             sys.stdout.flush()
#
#         last_percent_reported = percent
#
# def maybe_download(filename, expected_bytes, force=False):
#     """Download a file if not present, and make sure it's the right size."""
#     dest_filename = os.path.join(data_root, filename)
#     if force or not os.path.exists(dest_filename):
#         print('Attempting to download:', filename)
#         filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
#         print('\nDownload Complete!')
#     statinfo = os.stat(dest_filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified', dest_filename)
#     else:
#         raise Exception(
#             'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
#     return dest_filename
#
# train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
# test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# # wyd add to avoid download files again
# train_filename = os.path.join(data_root, 'notMNIST_large.tar.gz')
# test_filename = os.path.join(data_root, 'notMNIST_small.tar.gz')
#
# num_classes = 10
# np.random.seed(133)
#
# def maybe_extract(filename, force=False):
#     root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
#     if os.path.isdir(root) and not force:
#         # You may override by setting force=True.
#         print('%s already present - Skipping extraction of %s.' % (root, filename))
#     else:
#         print('Extracting data for %s. This may take a while. Please wait.' % root)
#         tar = tarfile.open(filename)
#         sys.stdout.flush()
#         tar.extractall(data_root)
#         tar.close()
#     data_folders = [
#         os.path.join(root, d) for d in sorted(os.listdir(root))
#         if os.path.isdir(os.path.join(root, d))]
#     if len(data_folders) != num_classes:
#         raise Exception(
#             'Expected %d folders, one per class. Found %d instead.' % (
#                 num_classes, len(data_folders)))
#     print(data_folders)
#     return data_folders
#
# train_folders = maybe_extract(train_filename)
# test_folders = maybe_extract(test_filename)

# wyd add to avoid extract files again
# train_folders = ['F:\\notMNIST_large/A', 'F:\\notMNIST_large/B', 'F:\\notMNIST_large/C', 'F:\\notMNIST_large/D',
#                  'F:\\notMNIST_large/E', 'F:\\notMNIST_large/F', 'F:\\notMNIST_large/G', 'F:\\notMNIST_large/H',
#                  'F:\\notMNIST_large/I', 'F:\\notMNIST_large/J']
# test_folders = ['F:\\notMNIST_small/A', 'F:\\notMNIST_small/B', 'F:\\notMNIST_small/C', 'F:\\notMNIST_small/D',
#                 'F:\\notMNIST_small/E', 'F:\\notMNIST_small/F', 'F:\\notMNIST_small/G', 'F:\\notMNIST_small/H',
#                 'F:\\notMNIST_small/I', 'F:\\notMNIST_small/J']
#
#
# image_size = 28  # Pixel width and height.
# pixel_depth = 255.0  # Number of levels per pixel.
#
# def load_letter(folder, min_num_images):
#     """Load the data for a single letter label."""
#     image_files = os.listdir(folder)
#     dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
#                          dtype=np.float32)
#     print(folder)
#     num_images = 0
#     for image in image_files:
#         image_file = os.path.join(folder, image)
#         try:
#             image_data = (ndimage.imread(image_file).astype(float) -
#                           pixel_depth / 2) / pixel_depth
#             if image_data.shape != (image_size, image_size):
#                 raise Exception('Unexpected image shape: %s' % str(image_data.shape))
#             dataset[num_images, :, :] = image_data
#             num_images = num_images + 1
#         except IOError as e:
#             print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
#
#     dataset = dataset[0:num_images, :, :]
#     if num_images < min_num_images:
#         raise Exception('Many fewer images than expected: %d < %d' %
#                         (num_images, min_num_images))
#
#     print('Full dataset tensor:', dataset.shape)
#     print('Mean:', np.mean(dataset))
#     print('Standard deviation:', np.std(dataset))
#     return dataset
#
#
# def maybe_pickle(data_folders, min_num_images_per_class, force=False):
#     dataset_names = []
#     for folder in data_folders:
#         set_filename = folder + '.pickle'
#         dataset_names.append(set_filename)
#         if os.path.exists(set_filename) and not force:
#             # You may override by setting force=True.
#             print('%s already present - Skipping pickling.' % set_filename)
#         else:
#             print('Pickling %s.' % set_filename)
#             dataset = load_letter(folder, min_num_images_per_class)
#             try:
#                 with open(set_filename, 'wb') as f:
#                     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
#             except Exception as e:
#                 print('Unable to save data to', set_filename, ':', e)
#
#     return dataset_names
#
#
# train_datasets = maybe_pickle(train_folders, 45000)
# test_datasets = maybe_pickle(test_folders, 1800)
#
#
# # pickle_file = train_datasets[1]  # index 0 should be all As, 1 = all Bs, etc.
# # with open(pickle_file, 'rb') as f:
# #     letter_set = pickle.load(f)  # unpickle
# #     sample_idx = np.random.randint(len(letter_set))  # pick a random image index
# #     sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
# #     plt.figure()
# #     plt.imshow(sample_image)  # display it
# #     plt.show()
#
#
# def make_arrays(nb_rows, img_size):
#     if nb_rows:
#         dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
#         labels = np.ndarray(nb_rows, dtype=np.int32)
#     else:
#         dataset, labels = None, None
#     return dataset, labels
#
#
# def merge_datasets(pickle_files, train_size, valid_size=0):
#     num_classes = len(pickle_files)
#     valid_dataset, valid_labels = make_arrays(valid_size, image_size)
#     train_dataset, train_labels = make_arrays(train_size, image_size)
#     vsize_per_class = valid_size // num_classes
#     tsize_per_class = train_size // num_classes
#
#     start_v, start_t = 0, 0
#     end_v, end_t = vsize_per_class, tsize_per_class
#     end_l = vsize_per_class + tsize_per_class
#     for label, pickle_file in enumerate(pickle_files):
#         try:
#             with open(pickle_file, 'rb') as f:
#                 letter_set = pickle.load(f)
#                 # let's shuffle the letters to have random validation and training set
#                 np.random.shuffle(letter_set)
#                 if valid_dataset is not None:
#                     valid_letter = letter_set[:vsize_per_class, :, :]
#                     valid_dataset[start_v:end_v, :, :] = valid_letter
#                     valid_labels[start_v:end_v] = label
#                     start_v += vsize_per_class
#                     end_v += vsize_per_class
#
#                 train_letter = letter_set[vsize_per_class:end_l, :, :]
#                 train_dataset[start_t:end_t, :, :] = train_letter
#                 train_labels[start_t:end_t] = label
#                 start_t += tsize_per_class
#                 end_t += tsize_per_class
#         except Exception as e:
#             print('Unable to process data from', pickle_file, ':', e)
#             raise
#
#     return valid_dataset, valid_labels, train_dataset, train_labels
#
#
# train_size = 200000
# valid_size = 10000
# test_size = 10000
#
# valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
#     train_datasets, train_size, valid_size)
# _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
#
# print('Training:', train_dataset.shape, train_labels.shape)
# print('Validation:', valid_dataset.shape, valid_labels.shape)
# print('Testing:', test_dataset.shape, test_labels.shape)
#
# def randomize(dataset, labels):
#     permutation = np.random.permutation(labels.shape[0])
#     shuffled_dataset = dataset[permutation,:,:]
#     shuffled_labels = labels[permutation]
#     return shuffled_dataset, shuffled_labels
# train_dataset, train_labels = randomize(train_dataset, train_labels)
# test_dataset, test_labels = randomize(test_dataset, test_labels)
# valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# pickle_file = os.path.join(data_root, 'notMNIST.pickle')
#
# try:
#     f = open(pickle_file, 'wb')
#     save = {
#         'train_dataset': train_dataset,
#         'train_labels': train_labels,
#         'valid_dataset': valid_dataset,
#         'valid_labels': valid_labels,
#         'test_dataset': test_dataset,
#         'test_labels': test_labels,
#         }
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#     f.close()
# except Exception as e:
#     print('Unable to save data to', pickle_file, ':', e)
#     raise
#
# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)

#
# import time
#
# def check_overlaps(images1, images2):
#     images1.flags.writeable=False
#     images2.flags.writeable=False
#     start = time.clock()
#     hash1 = set([hash(image1.tobytes()) for image1 in images1])
#     hash2 = set([hash(image2.tobytes()) for image2 in images2])
#     all_overlaps = set.intersection(hash1, hash2)
#     return all_overlaps, time.clock()-start
#
# r, execTime = check_overlaps(train_dataset, test_dataset)
# print('Number of overlaps between training and test sets: {}. Execution time: {}.'.format(len(r), execTime))
#
# r, execTime = check_overlaps(train_dataset, valid_dataset)
# print('Number of overlaps between training and validation sets: {}. Execution time: {}.'.format(len(r), execTime))
#
# r, execTime = check_overlaps(valid_dataset, test_dataset)
# print('Number of overlaps between validation and test sets: {}. Execution time: {}.'.format(len(r), execTime))


pickle_file = '/home/wyd/tensorflow_file/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

samples, width, height = train_dataset.shape
X_train = np.reshape(train_dataset,(samples,width*height))
y_train = train_labels

# Prepare testing data
samples, width, height = test_dataset.shape
X_test = np.reshape(test_dataset,(samples,width*height))
y_test = test_labels

# Instantiate（实例）
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)

# Fit
lg.fit(X_train, y_train)

# Predict
y_pred = lg.predict(X_test)

print(y_test)
print(y_pred)

# Score
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))