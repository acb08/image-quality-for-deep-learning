import time

import numpy as np
from hashlib import blake2b
from src.obj_det_analysis.analysis_tools import calculate_aggregate_results
from src.utils import detection_functions
from src.utils.shared_methods import _archive_processed_props, _check_extract_processed_props, \
    _get_processed_instance_props_path
from src.utils.vc import get_map_hash_mash
from src.obj_det_analysis.analysis_tools import get_instance_hash, electrons_to_dn


class ModelDistortionPerformanceResultOD:

    def __init__(self, dataset, result, convert_to_std, result_id, identifier=None, load_local=False,
                 manual_distortion_type_flags=None, report_time=False, force_recalculate=False,
                 distortions_ignore=(), special_extract_distortions=()):

        self.convert_to_std = convert_to_std
        self.result_id = result_id
        self.load_local = load_local
        self.manual_distortion_type_flags = manual_distortion_type_flags
        self.identifier = identifier
        self.dataset_id = result['test_dataset_id']
        self.report_time = report_time
        self._t0 = time.time()
        self._result = result
        self.model_artifact_id = self._result['model_artifact_id']
        self._force_recalculate = force_recalculate
        self.ignore_vc_hashes = False
        self.recalculation_completed = False  # allows double checking that recalculation has been performed
        self._distortions_ignore = tuple(distortions_ignore)
        self._special_extract_distortions = tuple(special_extract_distortions)
        self._distortions_ignore = self._distortions_ignore + self._special_extract_distortions
        self.vc_hash_mash = get_map_hash_mash()

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

        images = dataset['instances']['images']
        self.image_ids = detection_functions.get_image_ids(images)
        self.distortions = self.build_distortion_vectors(images)

        del images  # free up memory

        self.load_local = load_local
        if self.load_local:
            raise NotImplementedError

        # distortions_ignore helpful if not all distortion permutations present (e.g. when res and noise are coupled)
        if 'res' in self.distortions.keys() and 'res' not in distortions_ignore:
            self.res = self.distortions['res']
        else:
            self.res = np.ones(len(self.image_ids))
            # self.res = np.ones(len(image_ids))
            self.distortions['res'] = self.res

        if 'blur' in self.distortions.keys() and 'blur' not in distortions_ignore:
            self.blur = self.distortions['blur']
        else:
            self.blur = np.zeros(len(self.image_ids))
            self.distortions['blur'] = self.blur

        if 'noise' in self.distortions.keys() and 'noise' not in distortions_ignore:
            self.noise = self.distortions['noise']
        else:
            self.noise = np.zeros(len(self.image_ids))
            self.distortions['noise'] = self.noise

        if self.convert_to_std:
            self.noise = np.sqrt(self.noise)
            self.distortions['noise'] = self.noise

        self.distortion_space = self.get_distortion_space()

        self.image_id_map = self.map_images_to_dist_pts()
        # _parsed_mini_results = self.parse_by_dist_pt(result)

        self.shape = (len(self.distortion_space[0]), len(self.distortion_space[1]), len(self.distortion_space[2]))

        self.instance_hashes = {'predict': self.get_instance_hash()}

        self._3d_distortion_perf_props = None
        self.get_3d_distortion_perf_props()

        del self._result  # memory management after used

    def __len__(self):
        return len(self.image_ids)

    def __str__(self):
        if self.identifier:
            return str(self.identifier)
        else:
            return self.__repr__()

    def __repr__(self):
        return self.result_id

    def build_distortion_vectors(self, images):
        """
        Pull out distortion info from self._dataset['instances']['images'] and place in numpy vectors
        """
        distortions = {}
        try:
            for flag in self.distortion_type_flags:
                if flag not in self._distortions_ignore:
                    distortions[flag] = np.asarray([image[flag] for image in images])
        except TypeError:  # occurs when distortion_type_flags is None (i.e. on dataset without distortions)
            pass

        for flag in self._special_extract_distortions:
            if flag == 'noise':
                distortions[flag] = self.extract_pseudo_system_noise(images=images)
            else:
                raise NotImplementedError

        return distortions

    @staticmethod
    def extract_pseudo_system_noise(images, round_to_nearest=5):

        noise_electrons = np.asarray([image['snr']['estimated_post_adc_noise_electrons'] for image in images])
        well_depths = np.asarray([image['snr']['well_depth'] for image in images])
        noise = electrons_to_dn(noise_electrons, well_depths)

        noise = np.around(noise / round_to_nearest, 0) * round_to_nearest

        return noise

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

    def get_instance_hash(self):

        outputs = self._result['outputs']
        targets = self._result['targets']

        return get_instance_hash(outputs=outputs,
                                 targets=targets)

        # keys = list(outputs.keys())
        # keys.sort()
        # keys = keys[:num_use]
        #
        # data_mash = f'{len(outputs)}{len(targets)}'
        # for key in keys:
        #     output = str(outputs[key])
        #     target = str(targets[key])
        #     data_mash += f'{output}{target}'
        #
        # instance_hash = blake2b(data_mash.encode('utf-8')).hexdigest()
        #
        # return instance_hash

    def check_extract_processed_props(self, predict_eval_flag='predict'):
        return _check_extract_processed_props(self, predict_eval_flag=predict_eval_flag)

    def parse_by_dist_pt(self):

        """
        Maps images--predictions and targets--to distortion points

        *** Note, also handles the conversions of image ids to strings when they are used as keys in a dict logged in a .json
        file ***
        """

        parsed_mini_results = {}

        for dist_pt, image_ids in self.image_id_map.items():
            # parsed_outputs = {str(image_id): self._result['outputs'][str(image_id)] for image_id in image_ids}
            # parsed_targets = {str(image_id): self._result['targets'][str(image_id)] for image_id in image_ids}
            # trying with image_id left as an integer
            parsed_outputs = {image_id: self._result['outputs'][str(image_id)] for image_id in image_ids}
            parsed_targets = {image_id: self._result['targets'][str(image_id)] for image_id in image_ids}

            parsed_mini_results[dist_pt] = {'outputs': parsed_outputs, 'targets': parsed_targets}

        return parsed_mini_results

    def _time_string(self):
        return f'{round(time.time() - self._t0, 1)} s'

    def get_3d_distortion_perf_props(self, distortion_ids=('res', 'blur', 'noise'),
                                     details=False, make_plots=False, predict_eval_flag='predict'):
        """
        Shares name with equivalent method ModelDistortionPerformanceResult and CompositePerformanceResult but
        implemented differently since relevant performance metric is mAP rather than accuracy
        """

        if distortion_ids != ('res', 'blur', 'noise'):
            raise ValueError('method requires distortion_ids (res, blur, noise)')

        if not self._force_recalculate:

            if self._3d_distortion_perf_props is not None:
                return self._3d_distortion_perf_props

            cached_distortion_perf_props = self.check_extract_processed_props(predict_eval_flag=predict_eval_flag)

            if cached_distortion_perf_props:
                print('loaded cached processed props')
                self._3d_distortion_perf_props = cached_distortion_perf_props
                return self._3d_distortion_perf_props

        # code below this point in the method should only run once (i.e. in the __init__() block)
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

        nan_count = 0

        for i, res_val in enumerate(res_values):
            for j, blur_val in enumerate(blur_values):
                for k, noise_val in enumerate(noise_values):

                    dist_pt = (res_val, blur_val, noise_val)
                    mini_result = parsed_mini_results[dist_pt]

                    if len(mini_result['targets']) > 0:

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
                    else:
                        map3d[i, j, k] = np.NaN
                        full_extract[dist_pt] = 'empty'
                        nan_count += 1

        print('empty distortion point total: ', nan_count)

        parameter_array = np.asarray(parameter_array, dtype=np.float32)
        performance_array = np.atleast_2d(np.asarray(performance_array, dtype=np.float32)).T

        self._3d_distortion_perf_props = (res_values, blur_values, noise_values, map3d, parameter_array,
                                          performance_array, full_extract)

        self._force_recalculate = False  # recalculation completed/unneeded for get_3d_distortion_perf_props() call
        self.recalculation_completed = True
        print('mAP 3d calculated from dataset and test result')

        # call function here to log 3d distortion performance props
        self.archive_processed_props(res_values=res_values,
                                     blur_values=blur_values,
                                     noise_values=noise_values,
                                     perf_3d=map3d,
                                     distortion_array=parameter_array,
                                     perf_array=performance_array,
                                     predict_eval_flag=predict_eval_flag)

        return self._3d_distortion_perf_props

    def archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                perf_array, predict_eval_flag):
        print('archiving processed props')
        return _archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                        perf_array, predict_eval_flag,
                                        vc_hash_mash=self.vc_hash_mash)

    def get_processed_instance_props_path(self, predict_eval_flag='predict'):
        return _get_processed_instance_props_path(self, predict_eval_flag=predict_eval_flag)


class _PreProcessedDistortionPerformanceProps:

    def __init__(self, processed_props_path,
                 result_id,
                 outputs,
                 targets,
                 identifier=None,
                 predict_eval_flag='predict',
                 ignore_vc_hashes=False):
        self.processed_props_path = processed_props_path
        self.result_id = result_id
        self.identifier = identifier
        self.predict_eval_flag = predict_eval_flag
        self.outputs = outputs
        self.targets = targets
        self.vc_hash_mash = get_map_hash_mash()
        self.ignore_vc_hashes = ignore_vc_hashes
        self.instance_hashes = {'predict': self.get_instance_hash()}
        self._3d_distortion_perf_props = self.check_extract_processed_props()

    def get_processed_instance_props_path(self, predict_eval_flag=None):
        if predict_eval_flag is None:
            predict_eval_flag = self.predict_eval_flag
        return _get_processed_instance_props_path(self, predict_eval_flag=predict_eval_flag)

    def check_extract_processed_props(self, predict_eval_flag=None):
        if predict_eval_flag is None:
            predict_eval_flag = self.predict_eval_flag
        return _check_extract_processed_props(self, predict_eval_flag=predict_eval_flag)

    def get_3d_distortion_perf_props(self, distortion_ids=('res', 'blur', 'noise')):
        return self._3d_distortion_perf_props

    def get_instance_hash(self):
        return get_instance_hash(outputs=self.outputs, targets=self.targets)

    def __str__(self):
        if self.identifier:
            return str(self.identifier)
        else:
            return self.__repr__()

    def __repr__(self):
        return self.result_id





