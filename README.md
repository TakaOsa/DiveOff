# Discovering Multiple Solutions from a Single Task in Offline Reinforcement Learning

PyTorch implementation of Learning Diverse Behaviors in Offline RL (DiveOff). 
If you use our code or data please cite the paper.

We built our datasets based on [D4RL](https://github.com/Farama-Foundation/D4RL). 
Networks are trained using [PyTorch 1.13](https://github.com/pytorch/pytorch) and Python 3.8.15. 

### Usage
To train the DiveOff agent on walker2dvel-diverse-expert-v1, run
```
python training.py --policy DiveOff --env walker2dvel-diverse-expert-v1
```

For training on the antvel tasks, we set `--info_lr_rate 0.2`, e.g.,  
```
python train.py --env antvel-diverse-expert-medium-v1 --info_lr_rate 0.2
```

To visualize the behaviors learned by DiveOff on walker2dvel-diverse-expert-medium-v1, run
```
python visualize_diverse_policy.py --env walker2dvel-diverse-expert-medium-v1
```


### Bibtex
When you use our codes for your work, please cite:

```
@InProceedings{osa2024,
  title={Discovering Multiple Solutions from a Single Task in Offline Reinforcement Learning},
  author={Takayuki Osa and Tatsuya Harada},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year={2024}
}
```
