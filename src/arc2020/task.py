import numpy as np
from pathlib import Path
import json

from .mytypes import ImgMatrix, ImgPair, TestImgPair
from typing import List, Tuple


class Task(object):

    def __init__(self, task_path: Path):
        self.train = []  # type: List[ImgPair]
        self.test = []  # type: List[TestImgPair]
        self.tags = []  # type: List[Tuple[str, int]]
        self.name = task_path.stem
        self._append_task(task_path)

    def _append_task(self, task_path: Path):
        with open(str(task_path), 'r') as f:
            task = json.load(f)
        for cur_train in task['train']:
            self.train.append((ImgMatrix(cur_train['input']), ImgMatrix(cur_train['output'])))
        for cur_test in task['test']:
            test_input = ImgMatrix(cur_test['input'])
            if 'output' in cur_test:
                test_output = ImgMatrix(cur_test['output'])
            else:
                test_output = None
            self.test.append((test_input, test_output))
