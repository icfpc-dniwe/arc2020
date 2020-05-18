import numpy as np
from numba import njit
from scipy.special import expit
from typing import Tuple


def concatenate_features(feat):
    num_features = feat[0].shape[-1]
    feat = np.concatenate([cur_feat.reshape(-1, num_features) for cur_feat in feat], axis=0)
    return feat


def concatenate_targets(target):
    return np.concatenate([cur_target.ravel() for cur_target in target], axis=0)


@njit
def one_hot_color(color, num_colors=10):
    one_hot = np.zeros((num_colors,), dtype=np.int8)
    if color < one_hot.shape[0]:
        one_hot[color] = 1
    return one_hot


@njit
def one_hot_colors(colors, num_colors=10):
    one_hot = np.zeros((len(colors), num_colors), dtype=np.int8)
    for cur_idx, cur_color in enumerate(colors):
        if cur_color < one_hot.shape[1]:
            one_hot[cur_idx, cur_color] = 1
    return one_hot.ravel()


def count_all_colors(arr, num_colors=10):
    count_colors = np.zeros((num_colors,), dtype=np.int8)
    occurance_colors = np.zeros((num_colors,), dtype=np.int8)
    unique_colors, idx, counts = np.unique(arr, return_index=True, return_counts=True)
    occurance = np.argsort(idx)
    occurance_colors[:len(unique_colors)] = unique_colors[occurance][:num_colors]
    for cur_color, cur_count in zip(unique_colors, counts):
        if cur_color < num_colors:
            count_colors[cur_color] = cur_count
    return count_colors, occurance_colors


@njit
def gen_neighbor_indices(row_i, col_j, dist=1):
    indices = []
    for cur_row in range(row_i - dist, row_i + dist + 1):
        for cur_col in range(col_j - dist, col_j + dist + 1):
            indices.append((cur_row, cur_col))
    return indices


# @njit
def place_hist(cur_hist):
    np_hist = np.zeros((10,), dtype=np.float32)
    for cur_color, cur_counts in cur_hist.items():
        np_hist[cur_color] = cur_counts
    indices = np.argsort(np_hist)
    return indices.tolist() + np_hist[indices].tolist()


# @njit
def calc_symmetry_correct(submatrix):
    correct = (submatrix == np.flip(submatrix, axis=0)).astype(np.int32)
    correct = np.cumsum(correct, axis=0)
    height, width = correct.shape
    score = np.zeros((width, width, height), dtype=np.int32)
    correct = correct.T
    score[0, :, :] = correct
    for cur_w in range(1, width):
        score[cur_w, :-cur_w, :] = score[cur_w - 1, :-cur_w, :] + correct[cur_w:, :]
    return score.transpose((1, 2, 0))


# @njit
def axis_sym_correct(matrix, axis):
    if axis == 1:
        matrix = matrix.T
    height, width = matrix.shape
    score = np.zeros((height, width, height, width), dtype=np.int32)
    for cur_row in range(height):
        if cur_row > 0:
            score[cur_row, :, :-cur_row, :] = calc_symmetry_correct(matrix[cur_row:])
        else:
            score[cur_row, :, :, :] = calc_symmetry_correct(matrix)
    if axis == 1:
        score = score.transpose((1, 0, 3, 2))
    return score


count_matrix = np.ones((30, 30), dtype=np.float32)
count_matrix = np.cumsum(np.cumsum(count_matrix, axis=0), axis=1)
coeff_matrix = 1 / np.sqrt(count_matrix)


# @njit
def get_symmetry_matricies(matrix, topk=10, use_coeff=True):
    matrix_symmetry = np.array((axis_sym_correct(matrix, 0),
                                axis_sym_correct(matrix, 1)), dtype=np.int32)
    if use_coeff:
        height, width = matrix_symmetry.shape[1:3]
        cur_coeff = coeff_matrix[:height, :width][np.newaxis, np.newaxis, np.newaxis, ...]
        matrix_symmetry = matrix_symmetry * cur_coeff
    all_scores = matrix_symmetry.ravel()
    ravel_idx = np.argpartition(all_scores, -topk)[-topk:]
    ravel_idx = ravel_idx[np.argsort(all_scores[ravel_idx])]
    indices = np.unravel_index(ravel_idx, shape=matrix_symmetry.shape)
    #     print('scores', all_scores[ravel_idx])
    #     print('indices', indices)
    new_matricies = []
    for axis, row, col, height, width in zip(*indices):
        cur_matrix = matrix.copy()
        cur_matrix[row:row + height + 1, col:col + width + 1] = np.flip(
            matrix[row:row + height + 1, col:col + width + 1], axis=axis)
        new_matricies.append(cur_matrix)
    return new_matricies


# def show_symm_features(input_color):
#     features = get_symmetry_matricies(input_color, topk=5, use_coeff=True)
#     fig, axs = plt.subplots(1, len(features) + 1, figsize=(5 * (len(features) + 1), 5))
#     plot_one(axs[0], input_color, True, True)
#     for feature_idx, cur_feature in enumerate(features):
#         plot_one(axs[feature_idx + 1], cur_feature, True, False)


def im2freq(data):
    return np.fft.fft2(data)


def freq2im(f):
    return np.abs(np.fft.ifft2(f))


# @njit
def remmax(x):
    return x / x.max()


# @njit
def remmin(x):
    return x - np.amin(x, axis=(0, 1), keepdims=True)


# @njit
def touint8(x):
    return (remmax(remmin(x)) * (256 - 1e-4)).astype(np.uint8)


def to_complex(magnitudes, phases):
    return magnitudes * np.exp(1j * phases)


def fourier_features(input_color: np.ndarray):
    freq = im2freq(input_color)
    freq_log = im2freq(np.log(input_color + 1))
    #     img_freq = touint8(freq)
    autocorr = freq2im(freq * np.conj(freq))
    magnitudes = np.abs(freq) ** 2
    phases = np.angle(freq)
    more_corr = freq2im(to_complex(expit(magnitudes), phases))
    autocorr_log = freq2im(freq_log * np.conj(freq_log))
    magnitudes_log = np.abs(freq_log) ** 2
    phases_log = np.angle(freq_log)
    more_corr_log = freq2im(to_complex(expit(magnitudes_log), phases_log))
    return list(map(lambda x: x / np.max(x), [autocorr, touint8(autocorr), more_corr, touint8(more_corr),
                                              magnitudes, phases,
                                              autocorr_log, touint8(autocorr_log), more_corr_log,
                                              touint8(more_corr_log),
                                              magnitudes_log, phases_log]))


def make_features(input_color: np.ndarray,
                  local_neighb: int = 5,
                  max_dist: int = 7,
                  max_dilate: int = 3,
                  sym_max: int = 2
                  ) -> Tuple[np.ndarray, np.ndarray]:
    numeric_feat = []
    categoric_feat = []
    sym_max = min(sym_max, max_dist)
    # padding for feature calculation
    # f_features = fourier_features(input_color)
    sym_matricies = []  # get_symmetry_matricies(input_color)
    input_color = np.pad(input_color, max_dist, 'constant', constant_values=10)
    sym_matricies = [np.pad(cur_m, max_dist, 'constant', constant_values=10) for cur_m in sym_matricies]
    # f_features = np.pad(f_features, max_neighbors, 'constant', constant_values=0)
    nrows, ncols = input_color.shape
    for i in range(max_dist, nrows - max_dist):
        row_numeric_feat = []
        row_categoric_feat = []
        for j in range(max_dist, ncols - max_dist):
            min_i = max(i-local_neighb, 0)
            min_j = max(j-local_neighb, 0)
            cur_numeric_feat = []
            cur_categoric_feat = []
            cur_categoric_feat.extend([input_color[i][j]])
            cur_numeric_feat.extend([i, j, i + j])
            for cur_q in range(2, 16):
                cur_numeric_feat.extend([i % cur_q, j % cur_q, (i + j) % cur_q])
            cur_numeric_feat.extend([len(np.unique(input_color[i, :])),
                                     len(np.unique(input_color[:, j])),
                                     len(np.unique(input_color[min_i:i+local_neighb+1, min_j:j+local_neighb+1])),
                                     len(np.unique(np.diag(input_color, j-i))),
                                     len(np.unique(np.diag(np.fliplr(input_color), i-j)))])
            for dilate in range(1, max_dilate+1):
                cur_row = i
                cur_col = j
                diag = np.diag(input_color, cur_col-cur_row)
                counter_diag = np.diag(np.fliplr(input_color), cur_row-cur_col)
                rays = [input_color[cur_row, cur_col::dilate], input_color[cur_row, :cur_col+1:dilate],
                        input_color[cur_row::dilate, cur_col], input_color[:cur_row+1:dilate, cur_col],
                        diag[cur_col::dilate], diag[:cur_col+1:dilate],
                        counter_diag[cur_col::dilate], counter_diag[:cur_col+1:dilate]
                        ]
                rays += [input_color[max(cur_row-neighb_dist, 0):cur_row+neighb_dist+1:dilate,
                                     max(cur_col-neighb_dist, 0):cur_col+neighb_dist+1:dilate]
                         for neighb_dist in range(1, local_neighb + 1)]
                for cur_ray in rays:
                    num, cat = count_all_colors(cur_ray)
                    cur_numeric_feat.extend(num)
                    cur_categoric_feat.extend(cat)
            cur_categoric_feat.extend(input_color[i-max_dist:i+max_dist+1, j-max_dist:j+max_dist+1].ravel())
            for cur_m in sym_matricies:
                cur_categoric_feat.extend(cur_m[i-sym_max:i+sym_max+1, j-sym_max:j+sym_max+1].ravel())
            row_numeric_feat.append(cur_numeric_feat)
            row_categoric_feat.append(cur_categoric_feat)
        numeric_feat.append(row_numeric_feat)
        categoric_feat.append(row_categoric_feat)
    return np.array(numeric_feat, dtype=np.float32), np.array(categoric_feat, dtype=np.int32)
