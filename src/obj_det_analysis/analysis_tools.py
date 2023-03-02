import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.optimize
import random
from pathlib import Path

from matplotlib import pyplot as plt

import src.utils.definitions as definitions

# TODO: for images with no bounding boxes, my custom COCO(dataset) class adds a background annotation in the
#  __get_item__() method. If left unchanged, need to filter out background detections here

DEFAULT_IMAGE = np.ones((640, 640, 3))
DEFAULT_BBOX_COLOR = 'k'
BBOX_COLORS = {
    'TP': 'g',
    'TP_T': 'g',
    'FP': 'm',
    'FN': 'r',
    'gt': 'g',
    'pr': 'tab:cyan',
    'default': DEFAULT_BBOX_COLOR
}

DEFAULT_LINESTYLE = '-'
DEFAULT_LINESTYLE_PREDICT = '--'
BBOX_LINESTYLES = {
    'TP': DEFAULT_LINESTYLE_PREDICT,
    'TP_T': DEFAULT_LINESTYLE,
    'FP': DEFAULT_LINESTYLE_PREDICT,
    'FN': DEFAULT_LINESTYLE,
    'gt': DEFAULT_LINESTYLE,
    'pr': DEFAULT_LINESTYLE_PREDICT,
    'default': DEFAULT_LINESTYLE
}

DEFAULT_RANDOM_BOX_CONFIDENCE_LEVELS = (0.9, 0.7, 0.5, 0.3, 0.1)
DEFAULT_DETECT_CONFIDENCE = 0.95

NUM_CLASSES = 10


class SimulatedDataset:

    def __init__(self, num_images, num_boxes_range=(1, 4),
                 shape=DEFAULT_IMAGE.shape, width_range=(50, 350), aspect_ratio_range=(0.5, 2), max_overlap=0.1,
                 num_classes=NUM_CLASSES):
        self.num_images = num_images
        self.shape = shape
        self.width_range = width_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_boxes, self.max_boxes = num_boxes_range
        self.max_overlap = max_overlap
        self.image_ids = self.generate_image_ids()
        self.num_classes = num_classes

    def choose_num_bboxes(self):
        return random.randrange(self.min_boxes, self.max_boxes + 1)

    def generate_image_ids(self):
        return tuple([random.randrange(1, 100 * self.__len__()) for i in range(self.__len__())])

    def get_image_id(self, idx):
        return self.image_ids[idx]

    def generate_bbox_labels(self, num_bboxes):
        return [generate_random_label(num_classes=self.num_classes, exclude=None) for i in range(num_bboxes)]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        num_bboxes = self.choose_num_bboxes()
        boxes = random_boxes(image_shape=self.shape,
                             num_boxes=num_bboxes,
                             width_range=self.width_range,
                             aspect_ratio_range=self.aspect_ratio_range,
                             max_overlap=self.max_overlap)
        image_id = self.get_image_id(idx)
        labels = self.generate_bbox_labels(num_bboxes)

        return {'image_id': image_id, 'boxes': boxes, 'labels': labels}


def generate_random_label(num_classes=NUM_CLASSES, exclude=None):
    label = random.randrange(0, num_classes)
    if exclude is not None and label == exclude:
        return generate_random_label(num_classes=num_classes, exclude=exclude)
    return label


def bbox_to_rect(bbox, edgecolor=DEFAULT_BBOX_COLOR, linewidth=1, facecolor='none', linestyle='-'):

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    assert h >= 0 and w >= 0

    rect = patches.Rectangle((x0, y0), w, h, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor,
                             linestyle=linestyle)

    return rect


def check_box(box):
    if None in box:
        return False
    return True


def add_bboxes(bboxes, ax, bbox_labels=None, show=False):

    for i, bbox in enumerate(bboxes):
        if bbox_labels is not None:
            label = bbox_labels[i]
            if label in BBOX_COLORS.keys():
                edgecolor = BBOX_COLORS[label]
            else:
                edgecolor = DEFAULT_BBOX_COLOR
            if label in BBOX_LINESTYLES.keys():
                linestyle = BBOX_LINESTYLES[label]
            else:
                linestyle = DEFAULT_LINESTYLE
        else:
            edgecolor = DEFAULT_BBOX_COLOR
            linestyle = DEFAULT_LINESTYLE
        if check_box(bbox):  # screens out false negative [None, None, None, None] boxes
            rect = bbox_to_rect(bbox, edgecolor=edgecolor, linestyle=linestyle)
            ax.add_patch(rect)


def show_boxes(gt_boxes, predicted_boxes, image=DEFAULT_IMAGE, gt_labels=None, predict_labels=None, axes=None):

    gt_labels_default = len(gt_boxes) * ['gt']
    predict_labels_default = len(predicted_boxes) * ['pr']

    if axes is None:
        fig, (ax0, ax1) = plt.subplots(ncols=2)
        show = True
    else:
        ax0, ax1 = axes
        show = False

    ax0.imshow(image)
    add_bboxes(gt_boxes, ax0, bbox_labels=gt_labels_default)
    add_bboxes(predicted_boxes, ax0, bbox_labels=predict_labels_default)

    ax1.imshow(image)
    add_bboxes(gt_boxes, ax1, bbox_labels=gt_labels)
    add_bboxes(predicted_boxes, ax1, bbox_labels=predict_labels)

    if show:
        plt.tight_layout()
        plt.show()


def convert_gt_labels_to_str(gt_labels):
    if not set(gt_labels).issubset({0, 1}):
        raise ValueError('all elements of gt_labels must be either 0 or 1')
    converted_labels = ['TP_T' if _label == 1 else 'FN' for _label in gt_labels]
    return converted_labels


def convert_predict_labels_str(predict_labels):
    if not set(predict_labels).issubset({0, 1}):
        raise ValueError('all elements of gt_labels must be either 0 or 1')
    converted_labels = ['TP' if _label == 1 else 'FP' for _label in predict_labels]
    return converted_labels


def iou(b0, b1, verbose=False):

    """
    Calculates intersection of union for bounding boxes b0 and b1, where boxes are specified in xyxy format.
    """

    h0, w0 = b0[2] - b0[0], b0[3] - b0[1]
    h1, w1 = b1[2] - b1[0], b1[3] - b1[1]
    assert min(h0, w0, h1, w1) >= 0

    a0 = h0 * w0
    a1 = h1 * w1

    left_edge_intersect = max(b0[0], b1[0])
    right_edge_intersect = min(b0[2], b1[2])
    top_edge_intersect = max(b0[1], b1[1])  # indexed from top left corner
    bottom_edge_intersect = min(b0[3], b1[3])

    h_intersect = bottom_edge_intersect - top_edge_intersect
    h_intersect = max(0, h_intersect)
    w_intersect = right_edge_intersect - left_edge_intersect
    w_intersect = max(0, w_intersect)
    intersection = h_intersect * w_intersect

    intersection_coords = [left_edge_intersect, top_edge_intersect, right_edge_intersect, bottom_edge_intersect]

    union = a0 + a1 - intersection

    if verbose:
        print('b0 area: ', a0)
        print('b1 area: ', a1)
        print('intersection: ', left_edge_intersect, top_edge_intersect, right_edge_intersect, bottom_edge_intersect,
              ' area: ', intersection)
        print('iou: ', intersection / union)

    return intersection / union, intersection_coords


def check_overlap(existing_boxes, new_box, max_overlap):

    for box in existing_boxes:
        overlap, __ = iou(box, new_box)
        if overlap > max_overlap:
            return False

    return True


def random_boxes(image_shape=DEFAULT_IMAGE.shape, num_boxes=5, width_range=(50, 350), aspect_ratio_range=(0.5, 2),
                 max_overlap=0.35):

    x_range = (0, image_shape[1])
    y_range = (0, image_shape[0])

    boxes = []
    total_iters = 0
    max_iters = 20 * num_boxes

    while len(boxes) < num_boxes and total_iters < max_iters:

        total_iters += 1
        new_box = make_uniform_random_box(x_range=x_range,
                                          y_range=y_range,
                                          width_range=width_range,
                                          aspect_ratio_range=aspect_ratio_range)
        if valid_box(image_shape=image_shape, aspect_ratio_range=aspect_ratio_range, box=new_box):
            if max_overlap is not None:
                if check_overlap(boxes, new_box, max_overlap=max_overlap):
                    boxes.append(new_box)
            else:
                boxes.append(new_box)

    if len(boxes) != num_boxes:
        print(f'FYI: {len(boxes)} / {num_boxes} created')

    return boxes


def random_val_from_range(val_range):
    return val_range[0] + np.random.rand() * (val_range[1] - val_range[0])


def make_uniform_random_box(x_range, y_range, width_range, aspect_ratio_range):

    x0 = random_val_from_range(x_range)
    y0 = random_val_from_range(y_range)

    width = random_val_from_range(width_range)
    aspect_ratio = random_val_from_range(aspect_ratio_range)
    height = width * aspect_ratio

    x1 = x0 + width
    y1 = y0 + height

    return [x0, y0, x1, y1]


def simulate_detections(gt_boxes, gt_labels, class_accuracy=0.8, error_factor=0.25, image_shape=None,
                        num_classes=NUM_CLASSES, num_random_box_range=(5, 20),
                        confidence_levels=None, drop_detections=False):

    sim_detections = []
    scores = []
    predict_labels = []

    for i, gt_box in enumerate(gt_boxes):
        simulated_detection = make_anchored_box_with_gaussian_errors(anchor_box=gt_box,
                                                                     error_factor=error_factor,
                                                                     image_shape=image_shape)
        sim_detections.append(simulated_detection)
        confidence = np.random.rand()
        # confidence = min(confidence, 0.99)
        # confidence = max(confidence, 0)

        scores.append(confidence)

        gt_label = gt_labels[i]
        if np.random.rand() < class_accuracy:
            predicted_label = gt_label
        else:
            predicted_label = generate_random_label(num_classes=num_classes, exclude=gt_label)
        predict_labels.append(predicted_label)

    num_detections = len(sim_detections)
    if drop_detections and num_detections > 1:
        weights = 1 / (np.arange(num_detections) + 1)
        drop_options = np.arange(num_detections)
        num_drop = random.choices(drop_options, weights=weights, k=1)[0]
        if num_drop > 0:
            sim_detections = sim_detections[:-num_drop]
            scores = scores[:-num_drop]
            predict_labels = predict_labels[:-num_drop]

    num_random_boxes = random.randrange(num_random_box_range[0], num_random_box_range[1] + 1)

    if num_random_boxes > 0:
        if not image_shape:
            image_shape = DEFAULT_IMAGE.shape
        random_bboxes = random_boxes(image_shape=image_shape,
                                     num_boxes=num_random_boxes)
        random_labels = [generate_random_label(num_classes=num_classes) for i in range(num_random_boxes)]

        # mean_random_confidence = 0.6
        # random_confidence_levels = mean_random_confidence * np.ones(len(random_labels))
        # random_confidence_levels = random_confidence_levels + \
        #                            (1 - mean_random_confidence) * np.random.randn(len(random_labels))
        # random_confidence_levels = np.clip(random_confidence_levels, 0, 0.85)
        random_confidence_levels = np.random.rand(len(random_labels))
        random_confidence_levels = list(random_confidence_levels)
        scores.extend(random_confidence_levels)

        sim_detections.extend(random_bboxes)

        predict_labels.extend(random_labels)

    return sim_detections, scores, predict_labels


def gaussian_variation_from_start_point(start_point, error_factor, range_param, require_positive=True):

    new_val = start_point + np.random.randn() * error_factor * range_param

    if require_positive:
        if new_val > 0:
            return new_val
        else:
            return gaussian_variation_from_start_point(start_point, error_factor, range_param, require_positive=True)
    else:
        return start_point + np.random.randn() * error_factor * range_param


def generate_random_boxs(mean_count=5):

    num_random_boxes = np.random.poisson(mean_count)

    pass


def make_anchored_box_with_gaussian_errors(anchor_box, error_factor, image_shape=None):

    xa0, ya0, xa1, ya1 = anchor_box
    anchor_ctr_x, anchor_ctr_y = (xa0 + xa1) / 2, (ya0 + ya1) / 2

    anchor_width = xa1 - xa0
    anchor_height = ya1 - ya0

    x_ctr = gaussian_variation_from_start_point(anchor_ctr_x, error_factor=error_factor, range_param=anchor_width)
    y_ctr = gaussian_variation_from_start_point(anchor_ctr_y, error_factor=error_factor, range_param=anchor_height)

    width = gaussian_variation_from_start_point(anchor_width, error_factor=error_factor, range_param=anchor_width)
    height = gaussian_variation_from_start_point(anchor_height, error_factor=error_factor, range_param=anchor_height)

    x0, x1 = x_ctr - width / 2, x_ctr + width / 2
    y0, y1 = y_ctr - height / 2, y_ctr + height / 2

    if image_shape is not None:
        x_max = image_shape[1]
        y_max = image_shape[0]

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x_max, x1)
        y1 = min(y_max, y1)

    return [x0, y0, x1, y1]


def valid_box(image_shape, aspect_ratio_range, box):

    x0, x1 = box[0], box[2]
    y0, y1 = box[1], box[3]

    h = y1 - y0
    w = x1 - x0
    aspect_ratio = h / w

    x_min, x_max = (0, image_shape[1])
    y_min, y_max = (0, image_shape[0])

    if min(x0, x1) < x_min or max(x0, x1) > x_max:
        return False
    elif min(y0, y1) < y_min or max(y0, y1) > y_max:
        return False
    elif aspect_ratio < min(aspect_ratio_range) or aspect_ratio > max(aspect_ratio_range):
        return False
    else:
        return True


def match_bboxes(gt_boxes, predicted_boxes):
    """
    For a set of ground truth boxes (gt_boxes) and set of predicted boxes (predicted_boxes), this function uses linear
    sum assignment to find the best global correspondence between ground truth and predicted bounding boxes.

    Helpful gist for reference: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4

    :param gt_boxes: ground truth bounding boxes
    :param predicted_boxes: predicted bounding boxes
    :return:
        iou_values:

        gt:

        pr:

        predict_indices: column indices corresponding to the linear sum assignment that matches predicted bounding boxes
        to ground truth bounding boxes. predict_indices[i] gives the index corresponding to the best match
        predicted_box corresponding to the i-th ground truth bounding box. The
    """

    num_gt = len(gt_boxes)
    num_predicted = len(predicted_boxes)
    iou_matrix_size = max(num_gt, num_predicted)

    iou_matrix = np.zeros((iou_matrix_size, iou_matrix_size))

    for gt_idx, gt_box in enumerate(gt_boxes):
        for predict_idx, predicted_box in enumerate(predicted_boxes):
            iou_val, __ = iou(gt_box, predicted_box)
            iou_matrix[gt_idx, predict_idx] = iou_val

    gt_indices, predict_indices = scipy.optimize.linear_sum_assignment(cost_matrix=iou_matrix,
                                                                       maximize=True)

    iou_values = iou_matrix[gt_indices, predict_indices]
    gt = np.zeros(len(iou_values))
    # pr = np.zeros_like(iou_values)
    gt[:num_gt] = 1
    # pr[:num_predicted] = 1

    return iou_values, gt, predict_indices

    # return iou_matrix, gt_indices, predict_indices, gt_detect, predict_tp_fp


def get_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def get_recall(tp, fn):
    if tp + fn == 0:
        return 1
    return tp / (tp + fn)


def get_precision_recall(gt_det_scored, predict_det_scored):

    num_gt = len(gt_det_scored)
    num_predict = len(predict_det_scored)
    tp = np.sum(gt_det_scored)
    fp = num_predict - tp
    fn = num_gt - tp

    p = get_precision(tp, fp)
    r = get_recall(tp, fn)

    return p, r


def simulated_eval(pretend_dataset, error_factor, num_classes=NUM_CLASSES, image_shape=DEFAULT_IMAGE.shape,
                   confidence_levels=DEFAULT_RANDOM_BOX_CONFIDENCE_LEVELS,
                   drop_detections=True, class_accuracy=0.9):

    all_targets = {}
    all_outputs = {}

    for i in range(len(pretend_dataset)):
        pretend_data = pretend_dataset[i]
        gt_bboxes = pretend_data['boxes']
        gt_labels = pretend_data['labels']

        predicted_bboxes, scores, predicted_labels = simulate_detections(gt_bboxes, gt_labels,
                                                                         class_accuracy=class_accuracy,
                                                                         num_classes=num_classes,
                                                                         error_factor=error_factor,
                                                                         image_shape=image_shape,
                                                                         confidence_levels=confidence_levels,
                                                                         drop_detections=drop_detections)

        all_targets[pretend_data['image_id']] = pretend_data
        all_outputs[pretend_data['image_id']] = {'boxes': predicted_bboxes, 'scores': scores,
                                                 'labels': predicted_labels}

    return all_outputs, all_targets


def analyze_result(outputs, targets):

    processed_result = {}

    for image_id, target in targets.items():
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        prediction = outputs[image_id]
        predicted_boxes = prediction['boxes']

        iou_values, gt, predict_indices = match_bboxes(gt_boxes, predicted_boxes)


def pad_predict_boxes(predict_boxes, pad_length):

    predict_boxes = list(predict_boxes)
    pad_box = np.array([None, None, None, None])
    predict_boxes.extend(pad_length * [pad_box])

    return np.asarray(predict_boxes)


def visualize_single_image_result(ground_truth, prediction, iou_threshold=0.5, image_id=None, output_dir=None,
                                  image=DEFAULT_IMAGE):

    gt_boxes = np.asarray(ground_truth['boxes'])
    predicted_boxes = np.asarray(prediction['boxes'])

    scores = np.asarray(prediction['scores'])

    _detections_, _gt_mapped_scores_, _gt_, _predict_indices_ = assess_boxes(gt_boxes=gt_boxes,
                                                                             predict_boxes=predicted_boxes,
                                                                             scores=scores,
                                                                             iou_threshold=iou_threshold)

    if len(scores) < len(_gt_mapped_scores_):
        pad_length = len(_gt_mapped_scores_)
        predicted_boxes = pad_predict_boxes(predicted_boxes, pad_length)
        predicted_boxes = predicted_boxes[_predict_indices_]

    all_predicted_positives, true_positives = get_predict_true_pos_vectors(_gt_mapped_scores_,
                                                                           _detections_,
                                                                           threshold_score=np.min(_gt_mapped_scores_))

    _gt_ = np.asarray(_gt_)
    true_positives, false_negatives, false_positives = get_tp_fp_fn(gt=_gt_,
                                                                    true_positives=true_positives,
                                                                    all_predicted_positives=all_predicted_positives,
                                                                    return_sums=False)

    gt_box_labels = true_positives[:len(gt_boxes)]
    gt_check = gt_box_labels + false_negatives[:len(gt_boxes)]
    assert np.array_equal(gt_check, np.ones_like(gt_check))
    gt_box_labels = convert_gt_labels_to_str(gt_box_labels)

    vector_sum_check = true_positives + false_positives + false_negatives
    assert np.array_equal(vector_sum_check, np.ones_like(vector_sum_check))

    predict_box_labels = convert_predict_labels_str(true_positives)  # fn should map [None, None, None, None] boxes

    precision, recall = raw_pr_curves(_detections_, _gt_mapped_scores_, _gt_)
    precision_smoothed = precision_cleanup(precision)
    ap = get_average_precision(precision_smoothed, recall)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)

    show_boxes(gt_boxes=gt_boxes,
               predicted_boxes=predicted_boxes,
               gt_labels=gt_box_labels,
               predict_labels=predict_box_labels,
               axes=(ax0, ax1),
               image=image)

    ax2.plot(recall, precision)
    ax2.plot(recall, precision_smoothed)

    ax0.set_title('GT and Predict')
    ax1.set_title('Assessed')
    ax2.set_title(f'AP = {round(float(ap), 3)}')

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(Path(output_dir, f'{image_id}.png'))
    plt.show()

    total_boxes = len(_detections_)

    return precision, recall, total_boxes


def process_single_image_result(ground_truth, prediction, iou_threshold=0.5):

    sorted_gt_boxes = sort_by_label(ground_truth)
    sorted_predict_boxes = sort_by_label(prediction)

    unique_labels = set(sorted_gt_boxes.keys())
    unique_predict_labels = set(sorted_predict_boxes.keys())
    orphan_labels = unique_predict_labels.difference(unique_labels)

    unique_labels.update(unique_predict_labels)

    single_image_result = {}

    orphan_box_total = 0
    pad_total = 0
    total_boxes = 0

    for label in unique_labels:

        if label in sorted_gt_boxes.keys():
            gt_boxes = sorted_gt_boxes[label]['boxes']
            assert label not in orphan_labels
        else:
            gt_boxes = np.array([])
            assert label in orphan_labels

        if label in sorted_predict_boxes.keys():
            predict_boxes, scores = sorted_predict_boxes[label]['boxes'], sorted_predict_boxes[label]['scores']
        else:
            predict_boxes = np.array([])
            scores = np.array([])

        _detections_, _gt_mapped_scores_, _gt_, _predict_indices_ = assess_boxes(gt_boxes=gt_boxes,
                                                                                 predict_boxes=predict_boxes,
                                                                                 scores=scores,
                                                                                 iou_threshold=iou_threshold)

        _pad_length_ = len(_gt_mapped_scores_) - len(scores)
        _total_boxes_ = len(_detections_)

        # iou_values, gt, predict_indices = match_bboxes(gt_boxes=gt_boxes,
        #                                                predicted_boxes=predict_boxes)
        # total_boxes += len(iou_values)

        if len(scores) < len(_predict_indices_):
            pad_length = len(_predict_indices_) - len(scores)
            # scores = np.pad(scores, (0, pad_length))  # tuple (0, pad_length) pads only right side
            pad_total += pad_length

        # assert total_boxes == total_boxes
        #
        # gt_mapped_scores = scores[predict_indices]
        # detections = threshold_detections(iou_values, iou_threshold=iou_threshold)

        if label in orphan_labels:
            assert np.sum(_gt_) == 0
            orphan_box_total += len(_detections_)
            # assert len(_detections_) == len(_iou_values_)

        # assert list(detections) == _detections_
        # assert list(gt_mapped_scores) == _gt_mapped_scores_
        # assert list(gt) == _gt_

        single_image_result[label] = list(_detections_), list(_gt_mapped_scores_), list(_gt_)

    return single_image_result, orphan_box_total, pad_total, total_boxes


def assess_boxes(gt_boxes, predict_boxes, scores, iou_threshold):

    iou_values, gt, predict_indices = match_bboxes(gt_boxes=gt_boxes,
                                                   predicted_boxes=predict_boxes)

    if len(scores) < len(predict_indices):
        pad_length = len(predict_indices) - len(scores)
        scores = np.pad(scores, (0, pad_length))

    gt_mapped_scores = scores[predict_indices]
    detections = threshold_detections(iou_values, iou_threshold=iou_threshold)

    return list(detections), list(gt_mapped_scores), list(gt), predict_indices


def threshold_detections(iou_values, iou_threshold=0.5):

    detections = np.zeros_like(iou_values)
    detections[iou_values >= iou_threshold] = 1

    return detections


def raw_pr_curves(detections, gt_mapped_scores, gt):

    scores_unique = np.flip(np.unique(gt_mapped_scores))  # flip so that values are descending

    precision = [1]
    recall = [0]

    # __precision = [1]  # debug only
    # __recall = [0]  # debug only

    for i, threshold_score in enumerate(scores_unique):

        all_predicted_positives = np.zeros_like(gt_mapped_scores)
        all_predicted_positives[gt_mapped_scores >= threshold_score] = 1
        true_positives = all_predicted_positives * detections

        _all_predicted_positives_, _true_positives_ = get_predict_true_pos_vectors(gt_mapped_scores=gt_mapped_scores,
                                                                                   detections=detections,
                                                                                   threshold_score=threshold_score)

        assert np.array_equal(all_predicted_positives, _all_predicted_positives_)
        assert np.array_equal(true_positives, _true_positives_)

        tp, fn, fp = get_tp_fp_fn(gt=gt,
                                  true_positives=_true_positives_,
                                  all_predicted_positives=_all_predicted_positives_)

        p = get_precision(tp=tp, fp=fp)
        precision.append(p)
        r = get_recall(tp=tp, fn=fn)
        recall.append(r)
        #
        # __tp, __fn, __fp = get_tp_fp_fn(gt, true_positives, __masked_detections)
        # __p = get_precision(tp=__tp, fp=__fp)
        # __precision.append(__p)
        # __r = get_recall(tp=__tp, fn=__fn)
        # __recall.append(__r)

    if precision[0] == precision[1] and recall[0] == recall[1]:
        precision = precision[1:]
        recall = recall[1:]
        print('zero prune')

    if precision[-1] != 0 or recall[-1] != 1:
        precision.append(0)
        recall.append(1)
        # print('end append')

    # if __precision[0] == __precision[1] and __recall[0] == __recall[1]:
    #     __precision = __precision[1:]
    #     __recall = __recall[1:]
    #     print('zero prune')

    # if __precision[-1] != 0 or __recall[-1] != 1:
    #     __precision.append(0)
    #     __recall.append(1)

    # assert np.array_equal(precision, __precision)
    # assert np.array_equal(recall, __recall)

    return precision, recall  # , __precision, __recall


def get_predict_true_pos_vectors(gt_mapped_scores, detections, threshold_score):

    all_predicted_positives = np.zeros_like(gt_mapped_scores)
    all_predicted_positives[gt_mapped_scores >= threshold_score] = 1
    true_positives = all_predicted_positives * detections

    return all_predicted_positives, true_positives


def precision_cleanup(precision):

    smoothed_precision = []
    for i in range(len(precision)):
        smoothed_precision.append(max(precision[i:]))

    return smoothed_precision


def get_average_precision(smoothed_precision, recall):

    smoothed_precision = np.asarray(smoothed_precision)
    recall = np.asarray(recall)
    dr = recall[1:] - recall[:-1]
    assert min(dr) >= 0

    discrete_integrand = smoothed_precision[1:] * dr
    ap = np.sum(discrete_integrand)

    return ap


def get_tp_fp_fn(gt, true_positives, all_predicted_positives, return_sums=True):

    """

    :param gt: array indicating the presence of ground truth bounding boxes.  Indexing follows row index output order of
    the map_bboxes() function, which is always given by np.arange(n), where n = max(len(gt_boxes), len(predict_boxes)).
    :param true_positives: array indicating the presence of a predict box at or above the intersection-over-union
    threshold.
    :param all_predicted_positives:
    :param return_sums: bool, if True returns total number of true positives, false negatives, and false positives.
    Otherwise returns vectors.
    :return:
    """

    assert min(gt) >= 0 and max(gt) <= 1
    assert min(true_positives) >= 0 and max(true_positives) <= 1
    assert min(all_predicted_positives) >= 0 and max(all_predicted_positives) <= 1

    false_negatives = gt - true_positives
    false_positives = all_predicted_positives - gt
    false_positives = np.clip(false_positives, 0, np.inf)

    false_positive_check = all_predicted_positives - true_positives
    assert np.min(false_positive_check) >= 0
    assert np.sum(false_positive_check) >= np.sum(false_positives)

    if min(false_positives) < 0:
        print(false_positives)
        assert min(false_positives) >= 0

    if return_sums:

        tp = np.sum(true_positives)
        fn = np.sum(false_negatives)
        fp = np.sum(false_positives)

        return tp, fn, fp

    else:
        return true_positives, false_negatives, false_positives


def combine_results(single_image_results):

    result = {}

    total_boxes = 0

    for image_id, singe_image_result in single_image_results.items():

        for class_key, data in singe_image_result.items():

            __detections, __gt_mapped_scores, __gt = data
            total_boxes += len(__detections)

            if class_key not in result:
                result[class_key] = [], [], []

            detections, gt_mapped_scores, gt = result[class_key]
            detections.extend(__detections)
            gt_mapped_scores.extend(__gt_mapped_scores)
            gt.extend(__gt)

    return result, total_boxes


def sort_by_label(data):

    boxes = np.asarray(data['boxes'])
    labels = np.asarray(data['labels'])
    unique_labels = np.unique(labels)

    if 'scores' in data.keys():
        scores = np.asarray(data['scores'])
    else:
        scores = None

    sorted_box_data = {}

    if len(boxes) > 0:

        for label in unique_labels:

            filtered_boxes = boxes[labels == label, :]
            sorted_box_data[label] = {'boxes': filtered_boxes}

            if scores is not None:
                filtered_scores = scores[labels == label]
                sorted_box_data[label]['scores'] = filtered_scores

    return sorted_box_data




if __name__ == '__main__':

    _output_dir = Path(definitions.ROOT_DIR, 'map_demo')
    if not _output_dir.is_dir():
        Path.mkdir(_output_dir)

    _error_factor = 0.1
    _iou_threshold = 0.5
    _num_boxes_range = (1, 8)
    _num_images = 10
    _num_classes = 3

    _simulated_dataset = SimulatedDataset(num_images=_num_images,
                                          num_boxes_range=_num_boxes_range,
                                          num_classes=_num_classes)

    _all_outputs, _all_targets = simulated_eval(_simulated_dataset,
                                                num_classes=_num_classes,
                                                error_factor=_error_factor,
                                                image_shape=DEFAULT_IMAGE.shape,
                                                confidence_levels=DEFAULT_RANDOM_BOX_CONFIDENCE_LEVELS,
                                                drop_detections=False)

    # **************look at example "images" ********************

    _single_image_results = {}

    _single_image_ap_vals = []

    _total_boxes = 0
    _total_orphan_boxes = 0
    _pad_total = 0
    _total_boxes_triple_check = 0

    for _image_id in _all_targets.keys():

        _predict = _all_outputs[_image_id]
        # _predicted_boxes, _scores = _predict['boxes'], _predict['scores']

        _target = _all_targets[_image_id]
        _single_image_result, _orphan_boxes, _pad_num, _num_boxes = process_single_image_result(_target, _predict)
        # _total_orphan_boxes += _orphan_boxes
        # _pad_total += _pad_num
        # _total_boxes_triple_check += _num_boxes

        _single_image_results[_image_id] = _single_image_result

        _precision, _recall, _num = visualize_single_image_result(_target, _predict, output_dir=_output_dir,
                                                                  image_id=_image_id)
        # _total_boxes += _num
        #
        # _precision_smoothed = precision_cleanup(_precision)
        # _ap = get_average_precision(_precision_smoothed, _recall)
        # _single_image_ap_vals.append(_ap)

    _results, _total_boxes_check = combine_results(_single_image_results)

    _total_boxes_second_check = 0
    _class_precision_vals = []

    for _class_label, _data in _results.items():
        _detections, _gt_mapped_scores, _gt = _data
        _total_boxes_second_check += len(_detections)
        _precision, _recall = raw_pr_curves(_detections, _gt_mapped_scores, _gt)
        _precision_smoothed = precision_cleanup(_precision)
        _ap = get_average_precision(_precision_smoothed, _recall)
        _class_precision_vals.append(_ap)

        plt.figure()
        plt.plot(_recall, _precision_smoothed)
        plt.plot(_recall, _precision)
        plt.title(f'{_class_label}, {round(float(_ap), 3)}')
        plt.savefig(Path(_output_dir, f'class_{_class_label}.png'))
        plt.show()

    _mean_single_image_precision = np.mean(_single_image_ap_vals)
    _mean_avg_precision = np.mean(_class_precision_vals)

    # print('box totals:', _total_boxes, _total_boxes_check, _total_boxes_second_check, _total_boxes_triple_check)
    # print('orphan box total:', _total_orphan_boxes)
    # print('pad total:', _pad_total)
    # print('single image precision mean:', _mean_single_image_precision)
    # print('mAP:', _mean_avg_precision)


def calculate_aggregate_results(outputs, targets,
                                return_diagnostic_details=False, make_plots=True, output_dir_abs=None):

    single_image_results = {}

    for i, (image_id, target) in enumerate(targets.items()):
        predict = outputs[image_id]
        single_image_result, __, __, __ = process_single_image_result(target, predict)
        single_image_results[image_id] = single_image_result

    result, __ = combine_results(single_image_results)

    class_avg_precision_vals = []
    class_labels = []

    if return_diagnostic_details:
        recall_curves = []
        precision_curves = []
        precision_curves_smoothed = []

    else:
        recall_curves = None
        precision_curves = None
        precision_curves_smoothed = None

    for class_label, data in result.items():

        detections, gt_mapped_scores, gt = data
        precision, recall = raw_pr_curves(detections, gt_mapped_scores, gt)
        precision_smoothed = precision_cleanup(precision)
        ap = get_average_precision(precision_smoothed, recall)

        class_avg_precision_vals.append(ap)
        class_labels.append(class_label)

        if return_diagnostic_details:
            recall_curves.append(recall)
            precision_curves.append(precision)
            precision_curves_smoothed.append(precision_smoothed)

        if make_plots:
            plt.figure()
            plt.plot(recall, precision)
            plt.plot(recall, precision_smoothed)
            plt.title(f'{class_label}')
            if output_dir_abs is not None:
                plt.savefig(Path(output_dir_abs, f'{class_label}_pr.png'))
            plt.show()

    return class_labels, class_avg_precision_vals, recall_curves, precision_curves, precision_curves_smoothed
