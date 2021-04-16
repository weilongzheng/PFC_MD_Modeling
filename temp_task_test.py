"""Test task"""

import unittest
import numpy as np

from task import RikhyeTask


class TestTask(unittest.TestCase):

    def test_task(self):
        """"""
        n_time = 400
        dataset = RihkyeTask(Ntrain=100, Ntasks=2, blockTrain=True)
        input, target = dataset()
        assert input.shape == (n_time, 4)
        assert target.shape == (n_time, 2)

    def test_target(self):
        n_time = 400
        dataset = RihkyeTask(Ntrain=100, Ntasks=2, blockTrain=True)
        input, target = dataset()
        assert np.all(target.sum(axis=1) == 1)

    def test_task_switch(self):
        Ntrain = 10
        dataset = RihkyeTask(Ntrain=Ntrain, Ntasks=2, blockTrain=True)
        for i in range(Ntrain*2):
            input, target = dataset()
            if i < Ntrain:
                assert np.sum(input[0, :2]) == 1
                assert np.sum(input[0, 2:]) == 0
            else:
                assert np.sum(input[0, :2]) == 0
                assert np.sum(input[0, 2:]) == 1



if __name__ == '__main__':
    unittest.main()