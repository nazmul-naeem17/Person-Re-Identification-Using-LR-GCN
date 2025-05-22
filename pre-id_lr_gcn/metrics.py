import math
import numpy as np
import torch

from utils import standardize

def st_distribution(camera_ids, targets, frames, max_hist=3000,
                    eps=1e-7, interval=100.0):
    num_cams = len(np.unique(camera_ids))
    num_classes = len(np.unique(targets))
    spatial_temporal_sum = np.zeros((num_classes, num_cams))
    spatial_temporal_count = np.zeros((num_classes, num_cams))

    for cam, target, frame in zip(camera_ids, targets, frames):
        cam = cam - 1  # zero-based indexing for camera ids
        spatial_temporal_sum[target][cam] += frame
        spatial_temporal_count[target][cam] += 1

    spatial_temporal_avg = spatial_temporal_sum / (spatial_temporal_count + eps)

    distribution = np.zeros((num_cams, num_cams, max_hist))
    for i in range(num_classes):
        for j in range(num_cams - 1):
            for k in range(j + 1, num_cams):
                if spatial_temporal_count[i][j] == 0 or spatial_temporal_count[i][k] == 0:
                    continue
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                diff = abs(st_ij - st_ik)
                hist_ = int(diff / interval)
                distribution[j][k][hist_] += 1

    distribution = distribution / (distribution.sum(axis=2, keepdims=True) + eps)
    return distribution


def gaussian_func(x, mu, std):
    if std == 0:
        raise ValueError("In gaussian, std shouldn't be zero")
    temp1 = 1.0 / (std * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - mu, 2)) / (2 * math.pow(std, 2))
    return temp1 * np.exp(temp2)


def gaussian_kernel(length, mu=0, std=50):
    approximate_delta = 3 * std
    gaussian_vector = gaussian_func(np.arange(length), mu=mu, std=std)
    kernel = np.zeros((length, length))
    for i in range(length):
        k = 0
        for j in range(i, length):
            if k > approximate_delta:
                continue
            kernel[i][j] = gaussian_vector[j - i]
            k += 1
    kernel += kernel.transpose()
    for i in range(length):
        kernel[i][i] /= 2
    return kernel


def smooth_st_distribution(camera_ids, targets, frames, num_cams,
                           max_hist=3000, eps=1e-7, interval=100.0):

    distribution = st_distribution(camera_ids, targets, frames, max_hist, eps, interval)
    matrix = gaussian_kernel(distribution.shape[-1])

    for i in range(num_cams):
        for j in range(num_cams):
            distribution[i][j][:] = np.dot(matrix, distribution[i][j][:])

    distribution = distribution / (distribution.sum(axis=2, keepdims=True) + eps)
    return distribution


def joint_scores(query_features, query_cams, query_frames,
                 gallery_features,
                 gallery_cams, gallery_frames, distribution, alpha=5,
                 interval=100):

    query_features, gallery_features = standardize(query_features, gallery_features)
    scores = torch.Tensor()

    for feature, cam, frame in zip(query_features, query_cams, query_frames):
        feature_score = torch.matmul(gallery_features, feature)

        diff = torch.abs(gallery_frames - frame)
        hist_idx = (diff / interval).to(torch.int64)
        cam_idx = int(cam.item()) - 1  # zero-based camera index

        gallery_cam_indices = (gallery_cams - 1).to(torch.int64).tolist()

        # Safely index numpy array with Python integers
        st_probs_list = []
        for g_cam, h in zip(gallery_cam_indices, hist_idx.tolist()):
            g_cam_i = int(g_cam)
            h_i = int(h)
            st_probs_list.append(distribution[cam_idx][g_cam_i][h_i])

        st_score = torch.tensor(st_probs_list, device=query_features.device, dtype=torch.float32)

        score = 1 / (1 + torch.exp(-alpha * feature_score)) * 1 / (1 + 2 * torch.exp(-alpha * st_score))
        scores = torch.cat([scores, torch.unsqueeze(score, dim=0)])

    return scores


def AP_CMC(score, query_target, query_cam, gallery_targets, gallery_cams):
    index = np.argsort(score)
    # index = index[::-1]

    query_indices = np.argwhere(gallery_targets == query_target)
    camera_indices = np.argwhere(gallery_cams == query_cam)

    good_index = np.setdiff1d(query_indices, camera_indices, assume_unique=True)
    junk_index1 = np.argwhere(gallery_targets == -1)
    junk_index2 = np.intersect1d(query_indices, camera_indices)
    junk_index = np.append(junk_index2, junk_index1)

    average_precision = 0
    cmc = torch.IntTensor(len(index)).zero_()

    if good_index.size == 0:
        cmc[0] = -1
        return average_precision, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    if rows_good.size == 0:
        cmc[0] = -1
        return average_precision, cmc

    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        old_precision = i * 1.0 / rows_good[i] if rows_good[i] != 0 else 1.0
        average_precision += d_recall * (old_precision + precision) / 2

    return average_precision, cmc


def mAP(scores, query_targets, query_cams, gallery_targets, gallery_cams):
    num_queries = len(query_targets)
    CMC = torch.IntTensor(len(gallery_targets)).zero_()
    ap = 0.0

    for score, query_target, query_cam in zip(scores, query_targets, query_cams):
        ap_tmp, CMC_tmp = AP_CMC(score, query_target, query_cam, gallery_targets, gallery_cams)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    mean_ap = ap / num_queries
    CMC = CMC.float() / num_queries

    return mean_ap, CMC
