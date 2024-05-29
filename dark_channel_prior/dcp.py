import cv2
import numpy as np

from time import perf_counter
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import deltaE_ciede2000


def get_dark_image(img, ksize):
    min_channel_img = np.min(img, axis=2)

    # Minimum filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dark_img = cv2.erode(min_channel_img, kernel)

    return dark_img


def get_atmospheric_light(I, percent=0.001):
    dark_I = get_dark_image(I, 15)

    num = int(percent * dark_I.size)

    # The indices of the top num brightest pixels in dark_I
    indices = np.argsort(dark_I.ravel())[-num:]
    indices = np.unravel_index(indices, dark_I.shape)

    cand_A = I[indices]
    A = np.mean(cand_A, axis=0)  # Mean value
    # A = cand_A[np.mean(cand_A, axis=1).argmax()]  # Brightest value

    return A


def get_transmission(I, A, omega=0.95, t_min=0.1):
    t = 1 - omega * get_dark_image(I / A, 15)
    t = np.maximum(t, t_min)
    return t


def guided_filter(I, p, ksize, eps):
    mean_I = cv2.blur(I, (ksize, ksize))
    mean_p = cv2.blur(p, (ksize, ksize))
    corr_I = cv2.blur(I * I, (ksize, ksize))
    corr_Ip = cv2.blur(I * p, (ksize, ksize))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.blur(a, (ksize, ksize))
    mean_b = cv2.blur(b, (ksize, ksize))

    q = mean_a * I + mean_b

    return q


def refine_transmission(I, t):
    gray_I = cv2.cvtColor(np.uint8(I * 255), cv2.COLOR_BGR2GRAY)
    gray_I = gray_I.astype(np.float64) / 255

    t = guided_filter(gray_I, t, 60, 0.0001)

    return t


def dehaze(I):
    I = I.astype(np.float64) / 255

    A = get_atmospheric_light(I, 0.001)

    t = get_transmission(I, A, 0.95, 0.3)

    t = refine_transmission(I, t)

    J = (I - A) / t.reshape(t.shape[0], t.shape[1], 1) + A
    J = np.clip(J, 0, 1)

    return (J * 255).astype(np.uint8)


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

        img_dehazed = dehaze(img_hazy)

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

        img_dehazed = dehaze(img_hazy)

        cv2.namedWindow("result", 0)
        cv2.imshow("result", np.hstack((img_hazy, img_dehazed)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite(f"../images/collect/dehazed/{i}_dcp.{file_type[i-1]}", img_dehazed)
