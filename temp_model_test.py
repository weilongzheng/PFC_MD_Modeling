"""Test memory"""

import unittest
import numpy as np

from temp_model import PFC
from temp_model import MD
from temp_model import FullNetwork

class TestModel(unittest.TestCase):

    def test_pfc(self):
        """"""
        n_time = 200
        n_neuron = 1000
        n_neuron_per_cue = 200
        pfc = PFC(n_neuron,n_neuron_per_cue)
        input = np.random.randn(n_time, n_neuron)
        for input_t in input:
            output = pfc(input_t)
        assert output.shape == (n_neuron,)

    def test_md(self):
        """"""
        n_time = 200
        n_neuron = 1000
        Num_MD = 20
        num_active = 10 # num MD active per context
        md = MD(n_neuron, Num_MD, num_active)
        input = np.random.randn(n_time, n_neuron)
        for input_t in input:
            output = md(input_t)
        assert output.shape == (Num_MD,)
        
     def test_fullnetwork(self):
        n_time = 200
        n_neuron = 1000
        n_neuron_per_cue = 200
        Num_MD = 20
        num_active = 10 # num MD active per context
        pfc_md = FullNetwork(n_neuron,n_neuron_per_cue,Num_MD,num_active)
        input = np.random.randn(n_time, n_neuron)
        for input_t in input:
            pfc_md(input_t)
        


if __name__ == '__main__':
    unittest.main()