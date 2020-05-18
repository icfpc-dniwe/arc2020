import numpy as np
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from pathos.multiprocessing import Pool
from .features import make_features, concatenate_features
from ..histogram import get_target_hist, apply_color_map, inverse_color_map, calc_color_map
from ...operation import LearnableOperation
from ....classify import OutputSizeType
from .....mytypes import ImgMatrix, ImgPair


class LearnGBT(LearnableOperation):
    supported_outputs = [OutputSizeType.SAME, OutputSizeType.SQUARE_SAME]

    @staticmethod
    def _make_learnable_operation(use_aug: bool = False,
                                  residual_prediction: bool = True,
                                  use_hist: bool = True,
                                  max_dist: int = 7,
                                  max_dilate: int = 3,
                                  local_neighb: int = 5,
                                  sym_max: int = 2,
                                  n_estimators: int = 200,
                                  **gbt_kwargs):

        def prepare_features(matrix: ImgMatrix) -> np.ndarray:
            num_features, cat_features = make_features(matrix, local_neighb, max_dist, max_dilate, sym_max)
            features = np.concatenate((cat_features, num_features), axis=-1)
            return concatenate_features(features)

        def learn(imgs, targets):
            target_hist = get_target_hist(imgs)
            ensemble = XGBClassifier(n_estimators=n_estimators, n_jobs=-1, **gbt_kwargs)

            def worker(img_pair: ImgPair):
                input_img, output_img = img_pair
                if use_hist:
                    cur_color_map, residual = calc_color_map(input_img, target_hist)
                    input_img = apply_color_map(input_img, cur_color_map)
                    output_img = apply_color_map(output_img, cur_color_map)
                if residual_prediction:
                    diff = output_img != input_img
                    output_img = 10 * (1 - diff) + output_img * diff
                features = prepare_features(input_img)
                output = output_img.ravel()
                return features, output

            if use_aug:
                new_imgs = []
                new_targets = []
                for cur_img, cur_target in zip(imgs, targets):
                    for num_rot in range(4):
                        for transp in range(2):
                            for flip_axis in [0, 1, (1, 0)]:
                                p_aug = lambda x: np.rot90(np.flip(x, axis=flip_axis), k=num_rot)
                                if transp:
                                    aug = lambda x: np.transpose(p_aug(x))
                                else:
                                    aug = p_aug
                                new_imgs.append(aug(cur_img))
                                new_targets.append(aug(cur_target))
            else:
                new_imgs = imgs
                new_targets = targets
            with Pool(4) as p:
                results = p.map(worker, zip(new_imgs, new_targets))

            features = np.concatenate([cur_f for cur_f, cur_t in results], axis=0)
            targets = np.concatenate([cur_t for cur_f, cur_t in results], axis=0)
            feat_selector = VarianceThreshold()
            feat_selector.fit(features, targets)
            features = feat_selector.transform(features)
            ensemble.fit(features, targets)

            def predict(img: ImgMatrix
                        ) -> ImgMatrix:
                if use_hist:
                    cur_color_map, residual = calc_color_map(img, target_hist)
                    img = apply_color_map(img, cur_color_map)
                    inv_color_map = inverse_color_map(cur_color_map)
                cur_features = prepare_features(img)
                cur_features = feat_selector.transform(cur_features)
                preds = ensemble.predict(cur_features).reshape(img.shape)
                if residual_prediction:
                    diff_map = preds == 10
                    preds = img * diff_map + preds * (1 - diff_map)
                if use_hist:
                    preds = apply_color_map(preds, inv_color_map)
                return ImgMatrix(preds)

            return predict

        return learn
