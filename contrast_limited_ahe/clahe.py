import cv2
import numpy as np

from time import perf_counter
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import deltaE_ciede2000


def clahe(img, n_blocks=8, threshold=10.0):

    maps = get_local_maps(img, n_blocks, threshold)

    new_img = interpolate(img, maps)

    return new_img


def get_local_maps(img, n_blocks, threshold):
    h, w = img.shape
    block_h = int(h / n_blocks)
    block_w = int(w / n_blocks)

    # Split small regions and calculate the CDF for each
    maps = []
    for i in range(n_blocks):
        row_maps = []

        # Block border
        si, ei = i * block_h, (i + 1) * block_h
        if i == n_blocks - 1:
            ei = h

        for j in range(n_blocks):
            # Block border
            sj, ej = j * block_w, (j + 1) * block_w
            if j == n_blocks - 1:
                ej = w

            # Block image array
            block_img_arr = img[si:ei, sj:ej]

            # Calculate histogram and cdf
            hists = np.histogram(block_img_arr, bins=256, range=(0, 256))[0]
            clip_hists = clip_histogram(hists, threshold=threshold)  # clip histogram
            hists_cdf = calc_histogram_cdf(clip_hists)

            # Save
            row_maps.append(hists_cdf)
        maps.append(row_maps)

    return maps


def interpolate(img, maps):
    h, w = img.shape
    n_blocks = len(maps)
    block_h = int(h / n_blocks)
    block_w = int(w / n_blocks)

    # Interpolate every pixel using four nearest mapping functions
    new_img = np.empty_like(img)
    for i in range(h):
        for j in range(w):
            origin_val = img[i][j]

            r = int(
                np.floor(i / block_h - 0.5)
            )  # The row index of the left-up mapping function
            c = int(
                np.floor(j / block_w - 0.5)
            )  # The col index of the left-up mapping function

            x1 = (
                (i + 0.5) - (r + 0.5) * block_h
            ) / block_h  # The x-axis distance to the left-up mapping center
            y1 = (
                (j + 0.5) - (c + 0.5) * block_w
            ) / block_w  # The y-axis distance to the left-up mapping center

            # Four corners use the nearest mapping directly
            if r == -1 and c == -1:
                new_img[i][j] = maps[0][0][origin_val]
            elif r == -1 and c >= n_blocks - 1:
                new_img[i][j] = maps[0][-1][origin_val]
            elif r >= n_blocks - 1 and c == -1:
                new_img[i][j] = maps[-1][0][origin_val]
            elif r >= n_blocks - 1 and c >= n_blocks - 1:
                new_img[i][j] = maps[-1][-1][origin_val]
            # Four border case using the nearest two mapping
            elif r == -1 or r >= n_blocks - 1:
                if r == -1:
                    r = 0
                else:
                    r = n_blocks - 1
                left = maps[r][c][origin_val]
                right = maps[r][c + 1][origin_val]
                new_img[i][j] = (1 - y1) * left + y1 * right
            elif c == -1 or c >= n_blocks - 1:
                if c == -1:
                    c = 0
                else:
                    c = n_blocks - 1
                up = maps[r][c][origin_val]
                bottom = maps[r + 1][c][origin_val]
                new_img[i][j] = (1 - x1) * up + x1 * bottom
            # Bilinear interpolate for inner pixels
            else:
                lu = maps[r][c][origin_val]  # Mapping value of the left up cdf
                lb = maps[r + 1][c][origin_val]
                ru = maps[r][c + 1][origin_val]
                rb = maps[r + 1][c + 1][origin_val]
                new_img[i][j] = (1 - y1) * ((1 - x1) * lu + x1 * lb) + y1 * (
                    (1 - x1) * ru + x1 * rb
                )
    new_img = new_img.astype("uint8")
    return new_img


def calc_histogram_cdf(hists):
    """
    Calculate the CDF of the hists
    """
    hists_cumsum = np.cumsum(hists)
    const_a = 255 / np.sum(hists)
    hists_cdf = (const_a * hists_cumsum).astype("uint8")
    return hists_cdf


def clip_histogram(hists, threshold=10.0):
    """
    Clip the peak of histogram
    """
    all_sum = np.sum(hists)
    clip_limit = int(all_sum / 256 * threshold)

    total_extra = np.sum(np.maximum(hists - clip_limit, 0))
    mean_extra = int(total_extra / 256)

    clip_hists = np.minimum(hists + mean_extra, clip_limit)
    total_extra = all_sum - np.sum(clip_hists)

    k = 0
    while total_extra > 0:
        step_size = max(int(256 / total_extra), 1)
        for m in range(k, 256, step_size):
            if clip_hists[m] < clip_limit:
                clip_hists[m] += 1
                total_extra -= 1

                if total_extra == 0:
                    break

        k = (k + 1) % 256

    return clip_hists


def evaluate(dataset, n_files, hazy_img_type, gt_img_type, vr=50):
    mse_list = []
    psnr_list = []
    ssim_list = []
    ciede_list = []

    start = perf_counter()

    for i in range(1, n_files + 1):
        if dataset == "HazeRD":
            img_hazy = cv2.imread(f"../images/{dataset}/Hazy/{vr}/{i}.{hazy_img_type}")
        else:
            img_hazy = cv2.imread(f"../images/{dataset}/Hazy/{i}.{hazy_img_type}")

        h, w = img_hazy.shape[:2]
        new_h = min(h, 500)
        new_w = int(w * (new_h / h))
        img_hazy = cv2.resize(img_hazy, (new_w, new_h))

        (b, g, r) = cv2.split(img_hazy)  # Split channels

        new_b = clahe(b, threshold=4.0)
        new_g = clahe(g, threshold=4.0)
        new_r = clahe(r, threshold=4.0)

        img_dehazed = cv2.merge((new_b, new_g, new_r))  # Merge channels

        img_gt = cv2.imread(f"../images/{dataset}/GT/{i}.{gt_img_type}")
        img_gt = cv2.resize(img_gt, (new_w, new_h))

        mse = mean_squared_error(img_gt, img_dehazed)
        psnr = peak_signal_noise_ratio(img_gt, img_dehazed)
        ssim = structural_similarity(img_gt, img_dehazed, channel_axis=-1)

        img_dehazed_lab = cv2.cvtColor(img_dehazed, cv2.COLOR_BGR2LAB)
        img_gt_lab = cv2.cvtColor(img_gt, cv2.COLOR_BGR2LAB)
        ciede = deltaE_ciede2000(img_gt_lab, img_dehazed_lab, channel_axis=-1).mean()

        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        ciede_list.append(ciede)

    end = perf_counter()

    if dataset == "HazeRD":
        print(f"{dataset}数据集(visual range={vr})评估结果:")
    else:
        print(f"{dataset}数据集评估结果:")
    print(f"平均时间: {(end - start) / n_files}s")
    print(f"平均mse: {sum(mse_list) / n_files}")
    print(f"平均psnr: {sum(psnr_list) / n_files}")
    print(f"平均ssim: {sum(ssim_list) / n_files}")
    print(f"平均ciede: {sum(ciede_list) / n_files}")


if __name__ == "__main__":
    # evaluate("O-HAZE", 45, "jpg", "jpg")
    # evaluate("D-HAZY", 23, "bmp", "png")
    # evaluate("RESIDE", 50, "jpg", "png")

    # visual_ranges = [50, 100, 200, 500, 1000]
    # for vr in visual_ranges:
    #     evaluate("HazeRD", 15, "jpg", "jpg", vr=vr)

    file_type = ["jpg", "jpeg", "png", "jpeg", "png", "jpg", "jpg"]
    for i in range(1, 8):
        img_hazy = cv2.imread(f"../images/collect/{i}.{file_type[i-1]}")

        h, w = img_hazy.shape[:2]
        new_h = min(h, 500)
        new_w = int(w * (new_h / h))
        img_hazy = cv2.resize(img_hazy, (new_w, new_h))

        (b, g, r) = cv2.split(img_hazy)  # Split channels

        new_b = clahe(b, threshold=4.0)
        new_g = clahe(g, threshold=4.0)
        new_r = clahe(r, threshold=4.0)

        img_dehazed = cv2.merge((new_b, new_g, new_r))  # Merge channels

        cv2.namedWindow("result", 0)
        cv2.imshow("result", np.hstack((img_hazy, img_dehazed)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite(
        #     f"../images/collect/dehazed/{i}_clahe.{file_type[i-1]}", img_dehazed
        # )
