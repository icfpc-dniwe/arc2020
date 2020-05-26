import numpy as np
import cv2
from .ops.learnable import gbt
from .ops.operation import Transform
from .solver import Solver


def get_matrix_dims(inputs, targets=None):
    amatrix_dims={'in_matrix_height': [],
                  'in_matrix_width': [],
                  'out_matrix_height': [],
                  'out_matrix_width': []
                 }
    # iterate through training examples
    num_examples = len(inputs)
    ain_height = []
    ain_width = []
    aout_height = []
    aout_width = []
    for i in range(num_examples):
        input_image = np.array(inputs[i])
        in_matrix_height, in_matrix_width = input_image.shape
        ain_height.append(in_matrix_height)
        ain_width.append(in_matrix_width)
    amatrix_dims['in_matrix_height'].append(ain_height)
    amatrix_dims['in_matrix_width'].append(ain_width)
    if targets is not None:
        for i in range(num_examples):
            output_image = np.array(targets[i])
            out_matrix_height, out_matrix_width = output_image.shape
            aout_height.append(out_matrix_height)
            aout_width.append(out_matrix_width)
        amatrix_dims['out_matrix_height'].append(aout_height)
        amatrix_dims['out_matrix_width'].append(aout_width)
    return amatrix_dims


def get_matrix_rule(amatrix_dims):
    funcs_match_not_unknown = False
    multiplier_height = []
    multiplier_width = []
    addition_height = []
    addition_width = []
    answer_height = 'unknown' # if no rule found then uses size of 30
    height_param = 30
    answer_width = 'unknown'
    width_param = 30
    num_examples = len(amatrix_dims['in_matrix_width'][0])
    for i in range(num_examples):
        in_height = amatrix_dims['in_matrix_height'][0][i]
        out_height = amatrix_dims['out_matrix_height'][0][i]
        in_width = amatrix_dims['in_matrix_width'][0][i]
        out_width = amatrix_dims['out_matrix_width'][0][i]
        mult_height = out_height / in_height
        mult_width = out_width / in_width
        multiplier_height.append(mult_height)
        multiplier_width.append(mult_width)
        add_height = out_height - in_height
        addition_height.append(add_height)
        add_width = out_width - in_width
        addition_width.append(add_width)
    mult_height_unique = np.unique(multiplier_height)
    mult_width_unique = np.unique(multiplier_width)
    if len(mult_height_unique) == 1:
        answer_height = 'multiply by'
        height_param = mult_height_unique[0]
    if len(mult_width_unique) == 1:
        answer_width = 'multiply by'
        width_param = mult_width_unique[0]
    height_unique = np.unique(amatrix_dims['out_matrix_height'][0])
    width_unique = np.unique(amatrix_dims['out_matrix_width'][0])
    if answer_height != 'unknown' and answer_width == answer_height:
        funcs_match_not_unknown = True
    if len(height_unique) == 1 and funcs_match_not_unknown == False:
        answer_height = 'static'
        height_param = int(height_unique[0])
    if len(width_unique) == 1 and funcs_match_not_unknown == False:
        answer_width = 'static'
        width_param = int(width_unique[0])
    add_height_unique = np.unique(addition_height)
    add_width_unique = np.unique(addition_width)
    if answer_height != 'unknown' and answer_width == answer_height:
        funcs_match_not_unknown = True
    if len(add_height_unique) == 1 and funcs_match_not_unknown == False:
        answer_height = 'add this much'
        height_param = add_height_unique[0]
    if len(add_width_unique) == 1 and funcs_match_not_unknown == False:
        answer_width = 'add this much'
        width_param = add_width_unique[0]
    return answer_height, height_param, answer_width, width_param


def get_test_matrix_dims(amatrix_dims, matrix_rule):
    test_in_height = amatrix_dims['in_matrix_height'][0][0]
    test_in_width = amatrix_dims['in_matrix_width'][0][0]
    print('MR', matrix_rule)
    if matrix_rule[0] == 'static':
        test_out_height = matrix_rule[1]
    elif matrix_rule[0] == 'multiply by':
        test_out_height = test_in_height*matrix_rule[1]
    elif matrix_rule[0] == 'add this much':
        test_out_height = test_in_height + matrix_rule[1]
    else:
        test_out_height = -1
    if matrix_rule[2] == 'static':
        test_out_width = matrix_rule[3]
    elif matrix_rule[2] == 'multiply by':
        test_out_width = test_in_width*matrix_rule[3]
    elif matrix_rule[2] == 'add this much':
        test_out_width = test_in_width + matrix_rule[3]
    else:
        test_out_width = -1
    test_out_height = int(test_out_height)
    test_out_width = int(test_out_width)
    return test_out_height, test_out_width


class ResizeTransform(Transform):

    def __init__(self, rules, resize_max: bool = False):
        super().__init__()
        self.rules = rules
        self.out_size = None
        self.resize_max = resize_max

    def transform(self, img, target):
        input_rows, input_cols = img.shape
        target_rows, target_cols = target.shape
        if self.resize_max:
            new_img = np.zeros((30, 30), dtype=img.dtype)
            new_img[:input_rows, :input_cols] = img
            img = new_img
            new_target = np.zeros((30, 30), dtype=target.dtype)
            new_target[:target_rows, :target_cols] = target
            target = new_target
        else:
            max_rows = max(input_rows, target_rows)
            max_cols = max(input_cols, target_cols)
            img = cv2.resize(img, (max_cols, max_rows), interpolation=cv2.INTER_NEAREST)
            target = cv2.resize(target, (max_cols, max_rows), interpolation=cv2.INTER_NEAREST)
        return img, target

    def forward(self, img):
        amatrix_dims = get_matrix_dims([img])
        self.out_size = get_test_matrix_dims(amatrix_dims, self.rules)
        input_rows, input_cols = img.shape
        if self.resize_max:
            new_img = np.zeros((30, 30), dtype=img.dtype)
            new_img[:input_rows, :input_cols] = img
            img = new_img
        else:
            nrows = max(input_rows, self.out_size[0])
            ncols = max(input_cols, self.out_size[1])
            img = cv2.resize(img, (ncols, nrows), interpolation=cv2.INTER_NEAREST)
        return img

    def backward(self, preds):
        if self.resize_max:
            return preds[:self.out_size[0], :self.out_size[1]]
        else:
            return cv2.resize(preds, (self.out_size[1], self.out_size[0]), interpolation=cv2.INTER_NEAREST)


class SizeSolver(Solver):

    def __init__(self,
                 next_solver: Solver,
                 resize_max: bool = False):
        super().__init__()
        self.next_solver = next_solver
        self.resize_max = resize_max

    def solve(self):
        cur_pairs = self.task.train
        cur_imgs = [img for img, gt in cur_pairs]
        gt_imgs = [gt for img, gt in cur_pairs]
        amatrix_dims = get_matrix_dims(cur_imgs, gt_imgs)
        rules = get_matrix_rule(amatrix_dims)
        transform = rules[0] != 'unknown' or rules[2] != 'unknown'
        if transform:
            transformer = ResizeTransform(rules, self.resize_max)
            cur_pairs = [transformer.transform(img, gt) for img, gt in cur_pairs]
            self.task.train = cur_pairs
        ops = self.next_solver(self.task)
        if transform:
            ops = [transformer.forward] + ops + [transformer.backward]
        return ops
