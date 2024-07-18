# Recurrent Neural Networks of Mediodorsal Thalamus and Prefrontal Cortex in Temporal Contexts

## Model Framework
We develop recurrent neural networks with a thalamus-like component and synaptic plasticity rules to model the thalamocortical interactions in cognitive flexibility. We find that the MD component is able to extract context information by integrating context-relevant traces over trials and to suppress context-irrelevant neurons in the PFC. Incorporating the MD disjoints the contextual representations and enables efficient population coding in the PFC.

![alt text](https://github.com/weilongzheng/PFC_MD_Modeling/blob/main/MD_PFC.png?raw=true)

## Get started with training and testing
- Train a default network with *train_test_PFCMD_pytorch.py* for the cognitive task in [Rikhye et al. 2018](https://www.nature.com/articles/s41593-018-0269-z)

- Perform decoding analysis for context and rule with *decoding_analysis.py*

- Train a PFC-MD neural network on the [Neurogym](https://github.com/neurogym/neurogym) tasks with *run_PFCMD.py* in the *CL_neurogym* folder.
- The baseline continual learning methods, e.g., EWC and SI, are implemented in *run_baselines.py*.
- The model analysis is performed in *run_analysis.py*.

## Dependencies
The code is tested in Python 3.6 and Pytorch.

## Citation
If you use this code for your research, please cite our[paper:

```
@article{Zheng2024,
  title={Rapid Context Inference in a Thalamocortical Model Using Recurrent Neural Network},
  author={Wei-Long Zheng and Zhongxuan Wu and Ali Hummos and Guangyu Robert Yang and Michael M. Halassa},
  journal={Nature Communications},
  year={2024}
}
```

## Related Projects:
[PFC_MD_weights_stability](https://github.com/adityagilra/PFC_MD_weights_stability): Code for the computational model to avoid catastrophic forgetting as in Rikhye, Gilra and Halassa, Nature Neuroscience 2018

[ThalamusContextSwitchingCode](https://github.com/toxine4610/ThalamusContextSwitchingCode): Code base to recreate figures from Rikhye et al. Nature Neuroscience paper.
