"""
Tools to emulate an imager. Provides reimage() function as a drop in replacement for apply_distortions() function in
distort_coco_dataset.py, where the replacement simulates a sensor with a particular PSF and f-number.
"""

from src.pre_processing.distortion_tools import update_annotations


def reimage(image, distortion_functions, mapped_annotations, updated_image_id,
            remove_fragile_annotations=True, return_each_stage=False):

    pass
