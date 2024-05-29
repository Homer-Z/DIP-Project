import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2gray

from time import perf_counter
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.color import deltaE_ciede2000


def non_local_dehazing(img_hazy, air_light, gamma=1):
    h, w = img_hazy.shape[:2]

    img_hazy = img_hazy.astype(np.float64) / 255
    img_hazy_corrected = np.power(img_hazy, gamma)  # radiometric correction

    """
        Find Haze-lines
    """
    # Translate coordinate system to be air_light-centric
    dist_from_airlight = img_hazy_corrected - air_light

    # Calculate radius
    radius = np.linalg.norm(dist_from_airlight, axis=2)

    # Cluster the pixels to haze-lines
    # Use a KD-tree impementation for fast clustering according to their angles
    dist_unit_radius = dist_from_airlight.reshape(-1, 3)
    dist_norm = np.linalg.norm(dist_unit_radius, axis=1, keepdims=True)
    dist_unit_radius /= dist_norm + 1e-6
    n_points = 1000
    # Load pre-calculated uniform tessellation of the unit-sphere
    points = np.loadtxt(f"TR{n_points}.txt")
    mdl = KDTree(points)
    ind = mdl.query(dist_unit_radius)[1]

    """
        Estimating Initial Transmission
    """
    # Estimate radius as the maximal radius in each haze-line
    radius_max = np.bincount(ind.flatten(), radius.flatten(), minlength=n_points)
    # Handle cases where a haze-line has no points
    mask = radius_max == 0
    if np.any(mask):
        radius_max[mask] = np.max(radius)
    radius_new = radius_max[ind].reshape(radius.shape)

    # Estimate transmission as radii ratio
    transmission_estimation = radius / radius_new

    # Limit the transmission to the range [trans_min, 1] for numerical stability
    trans_min = 0.1
    transmission_estimation = np.clip(transmission_estimation, trans_min, 1)

    """
        Regularization
    """
    # Apply lower bound from the image
    trans_lower_bound = 1 - np.min(img_hazy / np.reshape(air_light, (1, 1, 3)), axis=2)
    transmission_estimation = np.maximum(transmission_estimation, trans_lower_bound)

    # Solve optimization problem
    # find bin counts for reliability - small bins (#pixels<50) do not comply with
    # the model assumptions and should be disregarded
    SMALL_BIN_THRESHOLD = 50
    bin_count = np.bincount(ind.flatten(), minlength=n_points)
    bin_count_map = bin_count[ind].reshape(h, w)
    bin_eval_fun = np.minimum(1, bin_count_map / SMALL_BIN_THRESHOLD)

    # Compute the standard deviation of radius, used as the data-term weight
    STD_LOWER_BOUND = 0.001
    STD_UPPER_BOUND = 0.1
    RADIUS_SCALE_FACTOR = 3

    radius_flat = radius.flatten()
    K_std = np.zeros(n_points)

    for i in range(n_points):
        mask = ind.flatten() == i
        if np.any(mask):
            K_std[i] = np.std(radius_flat[mask])

    radius_std = K_std[ind].reshape(h, w)
    radius_std_normalized = radius_std / np.max(radius_std)
    radius_eval_fun = np.minimum(
        1,
        RADIUS_SCALE_FACTOR
        * np.maximum(STD_LOWER_BOUND, radius_std_normalized - STD_UPPER_BOUND),
    )
    data_term_weight = bin_eval_fun * radius_eval_fun
    transmission = wls_optimization(
        transmission_estimation, data_term_weight, img_hazy, lambda_=0.1
    )

    """
        Dehazing
    """
    TRANSMISSION_MIN = 0.1
    LEAVE_HAZE_FACTOR = (
        1.06  # Leave a bit of haze for a natural look (set to 1 to reduce all haze)
    )
    ADJUST_PERCENT = [0.005, 0.995]

    air_light_reshaped = air_light.reshape(1, 1, 3)
    img_dehazed = (
        img_hazy_corrected
        - (1 - LEAVE_HAZE_FACTOR * transmission[..., np.newaxis]) * air_light_reshaped
    ) / np.maximum(transmission[..., np.newaxis], TRANSMISSION_MIN)

    img_dehazed = np.clip(img_dehazed, 0, 1)
    img_dehazed = np.power(img_dehazed, 1 / gamma)  # radiometric correction
    img_dehazed = adjust_contrast(img_dehazed, ADJUST_PERCENT)
    img_dehazed = (img_dehazed * 255).astype(np.uint8)

    return img_dehazed, transmission


def wls_optimization(input_image, data_weight, guidance, lambda_=0.1):
    small_num = 1e-5

    h, w = input_image.shape
    k = h * w

    guidance_gray = rgb2gray(guidance)

    # Compute horizontal and vertical affinities based on image gradients
    dy = np.diff(guidance_gray, axis=0)
    dy = -lambda_ / (np.abs(dy) ** 2 + small_num)
    dy = np.pad(dy, ((0, 1), (0, 0)), mode="constant").flatten("F")

    dx = np.diff(guidance_gray, axis=1)
    dx = -lambda_ / (np.abs(dx) ** 2 + small_num)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode="constant").flatten("F")

    # Construct a spatially inhomogeneous Laplacian matrix
    diagonal_indices = np.array([-h, -1])
    tmp = spdiags(np.vstack((dx, dy)), diagonal_indices, k, k)

    east = dx
    west = np.pad(dx, (h, 0), mode="constant")[:-h]
    south = dy
    north = np.pad(dy, (1, 0), mode="constant")[:-1]
    diagonal = -(east + west + south + north)

    Asmoothness = tmp + tmp.T + spdiags(diagonal, 0, k, k)

    # Normalize data weight
    data_weight -= np.min(data_weight)
    data_weight /= np.max(data_weight) + small_num

    # Adjust data weight and input based on reliability
    min_in_row = np.min(input_image, axis=0)
    reliability_mask = data_weight[0, :] < 0.6
    data_weight[0, reliability_mask] = 0.8
    input_image[0, reliability_mask] = min_in_row[reliability_mask]

    Adata = spdiags(data_weight.flatten("F"), 0, k, k)
    A = Adata + Asmoothness
    b = Adata * input_image.flatten("F")

    # Solve the linear system using the CSR-formatted matrix
    out = spsolve(csr_matrix(A), b).reshape(h, w, order="F")

    return out


def adjust_contrast(img, percentiles=[0.01, 0.99], contrast_factor=0.2):
    # Convert percentiles to actual percentile values
    low, high = np.percentile(
        img, [percentiles[0] * 100, percentiles[1] * 100], axis=(0, 1)
    )

    # Adjust the low and high thresholds
    high_adjusted = contrast_factor * high + (1 - contrast_factor) * np.maximum(
        high, np.mean(high)
    )
    low_adjusted = contrast_factor * low + (1 - contrast_factor) * np.minimum(
        low, np.mean(low)
    )

    # Clip and normalize the image
    img_clipped = np.clip(img, low_adjusted, high_adjusted)
    img_normalized = (img_clipped - low_adjusted) / (high_adjusted - low_adjusted)

    # Scale back to the original range
    img_rescaled = img_normalized * (high_adjusted - low_adjusted) + low_adjusted

    return img_rescaled


def get_dark_image(img, ksize):
    min_channel_img = np.min(img, axis=2)

    # Minimum filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dark_img = cv2.erode(min_channel_img, kernel)

    return dark_img


def estimate_airlight(I, percent=0.001):
    dark_I = get_dark_image(I, 15)

    num = int(percent * dark_I.size)

    indices = np.argsort(dark_I.ravel())[-num:]
    indices = np.unravel_index(indices, dark_I.shape)

    cand_A = I[indices]
    A = np.mean(cand_A, axis=0)  # Mean value
    # A = cand_A[np.mean(cand_A, axis=1).argmax()]  # Brightest value

    return A


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

        gamma = 0.6
        air_light = estimate_airlight(np.power(img_hazy / 255.0, gamma))
        img_dehazed, transmission_map = non_local_dehazing(img_hazy, air_light, gamma)

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

        gamma = 0.6
        air_light = estimate_airlight(np.power(img_hazy / 255.0, gamma))
        img_dehazed, transmission_map = non_local_dehazing(img_hazy, air_light, gamma)

        cv2.namedWindow("result", 0)
        cv2.imshow("result", np.hstack((img_hazy, img_dehazed)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite(
        #     f"../images/collect/dehazed/{i}_non_local.{file_type[i-1]}", img_dehazed
        # )
