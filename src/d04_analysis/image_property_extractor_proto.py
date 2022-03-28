# import numpy as np
# from scipy.stats import entropy
# from scipy import ndimage
# import os
# from PIL import Image
# import json
# from src.d01_pre_processing.pan_dist_proto import get_original_dataset_info
# import matplotlib.pyplot as plt
# import time
#
#
# def array_entropy(img, base=2):
#     """
#     :param img: input array (presumed to be an image)
#     :param base: base of entropy logarithm, default=2 for entropy in bits
#     :return: entropy of input array
#     """
#     img = np.asarray(img, dtype=np.uint8)
#     __, counts = np.unique(img, return_counts=True)
#
#     return entropy(counts, base=base)
#
#
# def grad_entropy(img):
#
#     img_x, img_y = gradients(img)
#     grad_hist = histogram_2d(img_x, img_y)
#     gradient_entropy = entropy_2d(grad_hist)
#
#     return 'grad', gradient_entropy
#
#
# def gradients(img, kernel_id='Sobel'):
#
#     if kernel_id == 'Sobel':
#         kernel = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#     elif kernel_id == 'Prewitt':
#         kernel = np.asarray([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
#     else:
#         return print('Invalid kernel ID')
#
#     img_x = ndimage.convolve(img, kernel, mode='nearest')
#     img_y = ndimage.convolve(img, kernel.T, mode='nearest')
#
#     return img_x, img_y
#
#
# def histogram_2d(img_x, img_y):
#
#     x_grads = np.ravel(img_x)
#     y_grads = np.ravel(img_y)
#     edges = np.arange(-255, 256)
#     grad_hist, __, __ = np.histogram2d(x_grads, y_grads, edges, density=True)
#
#     return grad_hist
#
#
# def entropy_2d(hist_img):
#
#     prob_hist = hist_img / np.sum(hist_img)
#     log_hist = np.copy(prob_hist)
#     log_hist[log_hist == 0] = 1
#     log_prob = -np.log2(log_hist)
#     _entropy = np.sum(prob_hist * log_prob)
#
#     return _entropy
#
#
# def fourier_magnitude_entropy(f_img, base=2):
#     f_img_mag = np.asarray(np.absolute(f_img), dtype=np.uint8)
#     return 'fourier_mag', array_entropy(f_img_mag)
#
#
# def fourier_2d_entropy(img_ft, base=2):
#
#     f_img_real = img_ft.real
#     f_img_imag = img_ft.imag
#     hist_img = histogram_2d(f_img_real, f_img_imag)
#     entropy_f_2d = entropy_2d(hist_img)
#
#     return 'fourier_2d', entropy_f_2d
#
#
# def fourier_entropies(img):
#
#     f_img = np.fft.fft2(img, axes=(0, 1))
#
#     f_mag_key, f_mag_ent = fourier_magnitude_entropy(f_img)
#     f_2d_key, f_2d_ent = fourier_2d_entropy(f_img)
#
#     return (f_mag_key, f_2d_key), (f_mag_ent, f_2d_ent)
#
#
# def shannon_entropy(img):
#     return 'shannon', array_entropy(img)
#
#
# def filenames_from_txt(file_path, start_line=0):
#
#     filenames = []
#     with open(file_path) as f:
#         for i, line in enumerate(f):
#             if i >= start_line:
#                 filenames.append(line.split(' ')[0])
#
#     return filenames
#
#
# def paths_from_tags(root_folder, tags, folder_only=False):
#
#     filename = 'metadata'
#     folder = root_folder
#     for tag in tags:
#         folder = os.path.join(folder, tag)
#         filename = filename + '_' + tag
#
#     filename = filename + '.json'
#
#     if not folder_only:
#         return folder, filename
#     else:
#         return folder
#
#
# def property_extractor(dictionary, target_key, iter_keys=None, traverse_keys=None):
#
#     values = []
#
#     if not traverse_keys:
#         traverse_keys = []
#     traverse_keys.append(target_key)
#
#     if not iter_keys:
#         iter_keys = list(dictionary.keys)
#
#     for iter_key in iter_keys:
#
#         inner_dict = dictionary[iter_key]
#         for sub_key in traverse_keys:
#             inner_dict = inner_dict[sub_key]
#         values.append(inner_dict)
#
#     return values
#
#
# def filename_from_tags(name_seed, tags, ext='.json'):
#
#     name = name_seed
#     for tag in tags:
#         name = name + '_' + tag
#     name = name + ext
#
#     return name
#
#
# if __name__ == '__main__':
#
#     original_dataset_tag = 'val_256'
#     image_tags = ['pan', 'r3_br2', 'b3']
#     img_mode = 'L'
#
#     early_stop = False
#     save_results = True
#
#     if early_stop == save_results:
#         print(f'FYI: early_stop = {early_stop} and save_results = {save_results}')
#
#     early_stop_num = 100
#     status_interval = 1000
#
#     original_folder, __ = get_original_dataset_info(original_dataset_tag)
#     image_folder, metadata_filename = paths_from_tags(original_folder, image_tags)
#     with open(os.path.join(image_folder, metadata_filename)) as f:
#         metadata = json.load(f)
#
#     names_labels = metadata['names_labels'].copy()
#     parent_image_dirs = metadata['parent_image_dirs'].copy()
#     image_info_ext = metadata['image_info'].copy()
#     del metadata  # to save memory and for fear of mutable zombie variables
#
#     # img = Image.open(os.path.join(image_folder, names_labels[0][0])).convert(img_mode)
#     property_funcs = [shannon_entropy, grad_entropy, fourier_entropies]
#
#     t0 = time.time()
#     image_names = []
#
#     for i, (img_name, img_label) in enumerate(names_labels):
#
#         image = Image.open(os.path.join(image_folder, img_name)).convert(img_mode)
#         image = np.asarray(image)
#
#         if early_stop and i >= early_stop_num:
#             break
#
#         image_names.append(img_name)
#
#         for property_function in property_funcs:
#
#             key, val = property_function(image)
#
#             if isinstance(key, str):
#                 image_info_ext[img_name][key] = val
#
#             elif isinstance(key, tuple):
#                 for j, inner_key in enumerate(key):
#                     image_info_ext[img_name][inner_key] = val[j]
#
#         if i % status_interval == 0:
#             print(f'{i} / {len(names_labels)}, {round(time.time() - t0)} seconds')
#
#     # shannon = property_extractor(image_info_ext,
#     #                              list(image_info_ext.keys()),
#     #                              ['shannon'])
#
#     shannon = property_extractor(image_info_ext, 'shannon', iter_keys=image_names)
#     grad = property_extractor(image_info_ext, 'grad', iter_keys=image_names)
#     fourier_2d = property_extractor(image_info_ext, 'fourier_2d', iter_keys=image_names)
#     fourier_mag = property_extractor(image_info_ext, 'fourier_mag', iter_keys=image_names)
#
#     plt.figure()
#     plt.plot(shannon)
#     plt.title('Shannon')
#     plt.show()
#
#     plt.figure()
#     plt.plot(grad)
#     plt.title('grad')
#     plt.show()
#
#     plt.figure()
#     plt.plot(fourier_mag)
#     plt.title('Fourier magnitude')
#     plt.show()
#
#     plt.figure()
#     plt.plot(fourier_2d)
#     plt.plot(grad)
#     plt.title('Fourier 2d')
#     plt.show()
#
#     save_name_seed = 'image_metadata_plus_properties'
#     save_name = filename_from_tags(save_name_seed, image_tags)
#
#     if save_results:
#         with open(os.path.join(image_folder, save_name), 'w') as f:
#             json.dump(image_info_ext, f)
#         print(f'{save_name} saved in {image_folder}')
