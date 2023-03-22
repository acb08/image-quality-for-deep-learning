import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.obj_det_analysis.analysis_tools import calculate_aggregate_results
# from src.utils.functions import _time_string
from src.utils import detection_functions
from src.utils import definitions
import time
# from src.utils.coco_functions import xywh_to_xyxy


class Sat6ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super(Sat6ResNet, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output 6 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)


class Sat6ResNet50(nn.Module):
    def __init__(self):
        super(Sat6ResNet50, self).__init__()

        self.model = models.resnet50(pretrained=True)

        # Change the output layer to output 6 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)


class Sat6DenseNet161(nn.Module):
    def __init__(self):
        super(Sat6DenseNet161, self).__init__()

        self.model = models.densenet161(pretrained=True)

        # Change the output layer to output 6 classes instead of 1000 classes
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)


class NumpyDataset(Dataset):

    def __init__(self, image_array, label_array, transform=None):
        self.label_array = label_array
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        label = self.label_array[idx]
        label = label
        if self.transform:
            image = self.transform(image)
        return image, label


class VariablePoissonNoiseChannelReplicated:
    """
    Selects Poisson noise standard deviation on sigma_poisson_range with uniform probability
    inclusive of the endpoints.

    generates zero-centered poisson noise with mean selected as described above, divides the noise by 255,
    adds to input tensor and clamps the output to fall on [0, 1]

    Copies the noise across input image channels so that a panchromatic image where the RGB channels are identical
    has the same noise added to each channel.

    """

    def __init__(self, sigma_poisson_range, clamp=True):
        self.low = sigma_poisson_range[0] ** 2
        self.high = sigma_poisson_range[1] ** 2 + 1  # add one because torch.randint excludes right endpoint
        self.clamp = clamp

    def __call__(self, tensor):
        # batch_size, channels, height, width = tensor.shape
        channels, height, width = tensor.shape
        noise_shape = (1, height, width)

        lambda_poisson = torch.randint(self.low, self.high, (1,))
        noise = (torch.poisson(torch.ones(noise_shape) * lambda_poisson).float() - lambda_poisson) / 255
        # channel_replicated_noise = noise.repeat(1, channels, 1, 1)
        channel_replicated_noise = noise.repeat(channels, 1, 1)
        noisy_tensor = tensor + channel_replicated_noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class VariableImageResize:
    """
    Builds a transform bank to re-size images and provides a method for randomly selecting the size of the
    transform to be used before being called.
    """

    def __init__(self, sizes, interpolation_mode='bilinear', antialias=False):
        self.sizes = sizes
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.transform_bank = self.build_transform_bank()
        self.size_keys = list(self.transform_bank.keys())

    def build_transform_bank(self):
        transform_bank = {}
        for size in self.sizes:
            size = int(size)
            if self.interpolation_mode == 'bilinear':
                new_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                                       antialias=self.antialias),
                     transforms.ToPILImage()]
                )
            elif self.interpolation_mode == 'bicubic':
                new_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=self.antialias),
                     transforms.ToPILImage()]
                )
            else:
                raise Exception('Invalid interpolation_mode')
            transform_bank[size] = new_transform
        print('build_transform_bank called')
        return transform_bank

    def get_size_key(self):
        """
        randomly returns one of the class's size keys, which can then be used in the __call__() method. Allows the user
        to know the resolution ahead of time for things like initializing arrays of the correct shape.
        """
        size_key = random.choice(self.size_keys)
        return size_key

    def __call__(self, image, size_key, dtype=np.uint8):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        transform_use = self.transform_bank[size_key]
        image = transform_use(image)
        image = np.asarray(image, dtype=dtype)
        return image


class VariableCOCOResize:
    """
    Builds a transform bank to re-size images and provides a method for randomly selecting the size of the
    transform to be used before being called.
    """

    def __init__(self, interpolation_mode='bilinear', antialias=False):
        # self.resolution_fractions = resolution_fractions
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias

    def __call__(self, image, res_frac, dtype=np.uint8):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        min_dimension = min(np.shape(image)[:2])
        new_size = int(res_frac * min_dimension)

        if self.interpolation_mode == 'bilinear':
            transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR,
                                           antialias=self.antialias),
                         transforms.ToPILImage()]
                    )
        elif self.interpolation_mode == 'bicubic':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BICUBIC,
                                   antialias=self.antialias),
                 transforms.ToPILImage()]
            )
        else:
            raise Exception('Invalid interpolation_mode')

        image = transform(image)

        return image


class VariableResolution:

    """
    Randomly selects a resize transform per the sizes argument passed in the __init__() method. Intended for use
    in a dataloader
    """

    def __init__(self, sizes, interpolation_mode='bilinear', antialias=False):
        self.sizes = sizes
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.transform_bank = self.build_transform_bank()

    def build_transform_bank(self):
        transform_bank = []
        for size in self.sizes:
            size = int(size)
            if self.interpolation_mode == 'bilinear':
                new_transform = transforms.Resize(size,
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=self.antialias)
            elif self.interpolation_mode == 'bicubic':
                new_transform = transforms.Resize(size,
                                                  interpolation=transforms.InterpolationMode.BICUBIC,
                                                  antialias=self.antialias)
            else:
                raise Exception('Invalid interpolation_mode')
            transform_bank.append(new_transform)
        print('build_transform_bank called')
        return transform_bank

    def __call__(self, tensor):
        transform_use = random.choice(self.transform_bank)
        return transform_use(tensor)


class PseudoArgs:
    """
    Imitates argparse.ArgumentParser() object to allow calls of functions.get_config(args) without actually using
    argparse (i.e. it's useful when a function needs to retrieve a config file)
    """
    def __init__(self, config_dir, config_name):
        self.config_dir = config_dir
        self.config_name = config_name


class COCO(Dataset):

    def __init__(self, image_directory, instances,
                 transform=transforms.Compose([transforms.ToTensor()]),
                 cutoff=None,
                 yolo_fmt=False):

        self.image_directory = Path(image_directory)
        if not self.image_directory.is_absolute():
            self.image_directory = Path(definitions.ROOT_DIR, self.image_directory)
        self.instances = instances
        self.images = self.instances['images']
        if cutoff is not None:
            self.images = self.images[:cutoff]
        self.annotations = self.instances['annotations']
        self.image_ids = detection_functions.get_image_ids(self.images)
        self.mapped_boxes_labels = detection_functions.map_boxes_labels(self.annotations, self.image_ids)
        self.transform = transform
        self.yolo_fmt = yolo_fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_data = self.images[idx]
        file_name = image_data['file_name']
        image_id = image_data['id']
        image = Image.open(Path(self.image_directory, file_name)).convert('RGB')
        image_annotations = self.mapped_boxes_labels[image_id]

        if len(image_annotations['boxes']) == 0:
            image_annotations = detection_functions.background_annotation(image)

        image_annotations['image_id'] = torch.tensor(image_id)

        if not self.yolo_fmt:
            image = self.transform(image)

        return image, image_annotations


class ModelDistortionPerformanceResultOD:

    def __init__(self, dataset, result, convert_to_std, result_id, identifier=None, load_local=False,
                 manual_distortion_type_flags=None, report_time=False):

        self.result = result
        self.dataset = dataset
        self.convert_to_std = convert_to_std
        self.result_id = result_id
        self.load_local = load_local
        self.manual_distortion_type_flags = manual_distortion_type_flags
        self.identifier = identifier
        self.dataset_id = result['test_dataset_id']
        self.report_time = report_time
        self._t0 = time.time()

        self.images = self.dataset['instances']['images']
        self._annotations = self.dataset['instances']['annotations']

        self.distortion_tags = dataset['distortion_tags']
        if 'distortion_type_flags' in dataset.keys():
            self.distortion_type_flags = dataset['distortion_type_flags']
            if manual_distortion_type_flags is not None:
                if set(self.distortion_type_flags) != set(manual_distortion_type_flags):
                    print(f'Warning: distortion type flags ({self.distortion_type_flags}) in dataset differ '
                          f'from manual distortion type flags ({manual_distortion_type_flags})')
        else:
            self.distortion_type_flags = manual_distortion_type_flags
        self.convert_to_std = convert_to_std
        self.load_local = load_local
        if self.load_local:
            raise NotImplementedError

        self.image_ids = detection_functions.get_image_ids(self.images)
        self.mapped_boxes_labels = detection_functions.map_boxes_labels(self._annotations, self.image_ids)

        self.distortions = self.build_distortion_vectors()

        if 'res' in self.distortions.keys():
            self.res = self.distortions['res']
        else:
            self.res = np.ones(len(self.image_ids))
            self.distortions['res'] = self.res

        if 'blur' in self.distortions.keys():
            self.blur = self.distortions['blur']
        else:
            self.blur = np.zeros(len(self.image_ids))
            self.distortions['blur'] = self.blur

        if 'noise' in self.distortions.keys():
            self.noise = self.distortions['noise']
        else:
            self.noise = np.zeros(len(self.image_ids))
            self.distortions['noise'] = self.noise

        if self.convert_to_std:
            self.noise = np.sqrt(self.noise)
            self.distortions['noise'] = self.noise

        self.distortion_space = self.get_distortion_space()

        self.image_id_map = self.map_images_to_dist_pts()
        self._parsed_mini_results = None

        self.shape = (len(self.distortion_space[0]), len(self.distortion_space[1]), len(self.distortion_space[2]))

        self._3d_distortion_perf_props = None

    def __len__(self):
        return len(self.image_ids)

    def __str__(self):
        if self.identifier:
            return str(self.identifier)
        else:
            return self.__repr__()

    def __repr__(self):
        return self.result_id

    def build_distortion_vectors(self):
        """
        Pull out distortion info from self._dataset['instances']['images'] and place in numpy vectors
        """
        distortions = {}
        for flag in self.distortion_type_flags:
            distortions[flag] = np.asarray([image[flag] for image in self.images])
        return distortions

    def map_images_to_dist_pts(self):

        if self.res is None or self.blur is None or self.noise is None:
            raise ValueError('')

        res_values, blur_values, noise_values = self.distortion_space

        id_vec = np.asarray(self.image_ids)

        image_id_map = {}

        for i, res_val in enumerate(res_values):
            res_inds = np.where(self.res == res_val)
            for j, blur_val in enumerate(blur_values):
                blur_inds = np.where(self.blur == blur_val)
                for k, noise_val in enumerate(noise_values):
                    noise_inds = np.where(self.noise == noise_val)
                    res_blur_inds = np.intersect1d(res_inds, blur_inds)
                    res_blur_noise_inds = np.intersect1d(res_blur_inds, noise_inds)

                    mapped_image_ids = id_vec[res_blur_noise_inds]

                    image_id_map[(res_val, blur_val, noise_val)] = mapped_image_ids

        return image_id_map

    def get_distortion_space(self):
        return np.unique(self.res), np.unique(self.blur), np.unique(self.noise)

    def get_distortion_matrix(self):
        pass

    def parse_by_dist_pt(self):

        if self._parsed_mini_results is None:

            parsed_mini_results = {}

            for dist_pt, image_ids in self.image_id_map.items():
                parsed_outputs = {str(image_id): self.result['outputs'][str(image_id)] for image_id in image_ids}
                parsed_targets = {str(image_id): self.result['targets'][str(image_id)] for image_id in image_ids}

                parsed_mini_results[dist_pt] = {'outputs': parsed_outputs, 'targets': parsed_targets}

            # TODO: figure out why image ids are stored as strings in result['outputs] and result['targets']

            self._parsed_mini_results = parsed_mini_results

        return self._parsed_mini_results

    def _time_string(self):
        return f'{round(time.time() - self._t0, 1)} s'

    def get_3d_distortion_perf_props(self, distortion_ids, details=False, make_plots=False, force_recalculate=False):

        if not force_recalculate and self._3d_distortion_perf_props is not None:
            return self._3d_distortion_perf_props

        if distortion_ids != ('res', 'blur', 'noise'):
            raise ValueError('method requires distortion_ids (res, blur, noise)')

        if self.report_time:
            print(f'getting 3d distortion perf probs, {self._time_string()}')

        parsed_mini_results = self.parse_by_dist_pt()

        if self.report_time:
            print(f'parsed mini results, {self._time_string()}')

        res_values, blur_values, noise_values = self.get_distortion_space()

        map3d = np.zeros(self.shape, dtype=np.float32)
        parameter_array = []  # for use in curve fits
        performance_array = []  # for use in svd
        full_extract = {}

        for i, res_val in enumerate(res_values):
            for j, blur_val in enumerate(blur_values):
                for k, noise_val in enumerate(noise_values):

                    dist_pt = (res_val, blur_val, noise_val)
                    mini_result = parsed_mini_results[dist_pt]

                    processed_results = calculate_aggregate_results(outputs=mini_result['outputs'],
                                                                    targets=mini_result['targets'],
                                                                    return_diagnostic_details=details,
                                                                    make_plots=make_plots)

                    class_labels, class_avg_precision_vals, recall, precision, precision_smoothed = processed_results
                    mean_avg_precision = np.mean(class_avg_precision_vals)
                    map3d[i, j, k] = mean_avg_precision
                    parameter_array.append([res_val, blur_val, noise_val])
                    performance_array.append(mean_avg_precision)
                    full_extract[dist_pt] = processed_results

        parameter_array = np.asarray(parameter_array, dtype=np.float32)
        performance_array = np.atleast_2d(np.asarray(performance_array, dtype=np.float32)).T

        self._3d_distortion_perf_props = (res_values, blur_values, noise_values, map3d, parameter_array,
                                          performance_array, full_extract)

        return self._3d_distortion_perf_props
