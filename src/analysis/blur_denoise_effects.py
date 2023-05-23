import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.poisson_sim import initial_electrons_and_image, apply_blur
from PIL import Image
import matplotlib


def denoise_raw(fill_fractions, blur_stds, output_dir):

    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    data = {}

    for fill_fraction in fill_fractions:

        initial_electrons, initial_image = initial_electrons_and_image(fill_fraction=fill_fraction)
        stds = []
        snrs = []

        ff_string = f'{int(fill_fraction * 100)}-percent-fill'
        sub_dir = Path(output_dir, ff_string)

        if not sub_dir.is_dir():
            sub_dir.mkdir()
            initial_image.save(Path(sub_dir, 'initial_image.png'))

        with open(Path(sub_dir, 'stats.txt'), 'w') as log_file:

            for i, blur_std in enumerate(blur_stds):

                if blur_std > 0:
                    blurred_image = apply_blur(initial_image, std=blur_std)
                else:
                    blurred_image = copy.deepcopy(initial_image)

                histogram(image=blurred_image, output_dir=sub_dir, idx=i)

                std = np.std(blurred_image)
                mean = np.mean(blurred_image)
                if std > 0:
                    snr = mean / std
                else:
                    snr = -1
                stds.append(std)
                snrs.append(snr)

                blurred_image.save(Path(sub_dir, f'blurred_{i}.png'))

                log_stats(image=blurred_image,
                          mean=mean,
                          img_std=std,
                          snr=snr,
                          idx=i,
                          blur_std=blur_std,
                          file=log_file)

            snr_plot(stds=stds,
                     snrs=snrs,
                     blur_stds=blur_stds,
                     output_dir=sub_dir)

        data[fill_fraction] = stds

    return data


def log_stats(image, mean, img_std, snr, idx, blur_std, file=None):

    min_val, max_val, median = np.min(image), np.max(image), np.median(image)

    print(f'image {idx}, sigma blur = {blur_std}:', file=file)
    print(f'min / max: {min_val} / {max_val}', file=file)
    print(f'mean / median: {mean} / {median}', file=file)
    print(f'std: {img_std}', file=file)
    if snr < 0:
        snr = 'NaN'
    print(f'snr: {snr}', '\n', file=file)


def snr_plot(stds, snrs, blur_stds, output_dir):

    stds = np.array(stds)
    snrs = np.array(snrs)
    blur_stds = np.array(blur_stds)

    valid_snr_indices = np.where(snrs >= 0)
    snrs_plot = snrs[valid_snr_indices]
    snr_plot_blur_stds = blur_stds[valid_snr_indices]

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

    ax0.plot(blur_stds, stds)
    ax0.set_xlabel(r'$\sigma$-blur')
    ax0.set_ylabel('std (DN)')

    ax1.plot(snr_plot_blur_stds, snrs_plot)
    ax1.set_xlabel(r'$\sigma$-blur')
    ax1.set_ylabel('SNR')

    plt.savefig(Path(output_dir, f'snr_plot.png'))
    plt.close()


def histogram(image, output_dir=None, idx=0):

    bins = np.unique(image)
    bins_ext = np.zeros(len(bins) + 1)
    bins_ext[:-1] = bins
    bins_ext[-1] = np.max(bins + 1)

    hist, bin_edges = np.histogram(image, bins=bins_ext)

    plt.figure()
    plt.stairs(hist, bin_edges)
    if output_dir is not None:
        plt.savefig(Path(output_dir, f'hist-{idx}.png'))
    plt.close()


if __name__ == '__main__':

    _blur_stds = np.linspace(0, 5, num=26, endpoint=True)

    _noise_data = denoise_raw(fill_fractions=[0.3, 0.4, 0.5, 0.6, 0.7], blur_stds=_blur_stds,
                              output_dir='/home/acb6595/coco/analysis/poisson_sim/raw-manual')

    plt.figure()
    for _fill_frac, _stds in _noise_data.items():
        plt.plot(_blur_stds, _stds, label=_fill_frac)
    plt.legend()
    plt.show()

