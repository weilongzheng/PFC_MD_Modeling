"""Test memory"""

import unittest
import numpy as np

from temp_model import PFC

class TestModel(unittest.TestCase):

    def test_pfc(self):
        """"""
        n_time = 200
        n_neuron = 1000
        pfc = PFC(n_neuron)
        input = np.random.randn(n_time, n_neuron)
        for input_t in input:
            output = pfc(input_t)
        assert output.shape == (n_neuron,)




if __name__ == '__main__':
    unittest.main()