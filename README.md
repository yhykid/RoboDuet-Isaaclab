# RoboDuet: Whole-body Legged Loco-Manipulation with Cross-Embodiment Deployment

# 没写完 刚开始 
<img src="./media/cross.png" alt="cross" width="100%" style="margin-top: -90px;">

<h3> <a href="https://github.com/locomanip-duet/locomanip-duet.github.io/blob/master/RoboDuet.pdf">📝 Paper</a> | <a href="https://locomanip-duet.github.io/"> 🖼️ Project Page</a></h3>

This repo is an official PyTorch implementation of our paper *<b>"RoboDuet: Whole-body Legged Loco-Manipulation with Cross-Embodiment Deployment"</b>*. Thanks to the cooperative policy mechanism and two-stage training strategy, the proposed framework demonstrates agile whole-body control and cross-embodiment deployment capabilities. <b>📺️More demo details can be found on our project page.</b>

# [Deployments](https://github.com/locomanip-duet/RoboDuet_Deployment)
We will provide deployment code for both the [Unitree Go1 EDU]() and [Unitree Go2 EDU]() robots mounted with [ARX5](). Additionally, we support using the [Meta Quest 3](https://www.meta.com/quest/quest-3/) to control the end-effector pose of the ARX.

**Please visit [RoboDuet-Deployment](https://github.com/locomanip-duet/RoboDuet_Deployment) for more details.**

</br>

# Acknowledgement
The base implementation is largely borrowed from [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways), an impressive work that demonstrates robust locomotion with a multiplicity of behaviors (MoB). We are deeply grateful for their contribution to the open-source community.

</br>

# Citation
```
@misc{pan2024roboduetwholebodyleggedlocomanipulation,
      title={RoboDuet: Whole-body Legged Loco-Manipulation with Cross-Embodiment Deployment}, 
      author={Guoping Pan and Qingwei Ben and Zhecheng Yuan and Guangqi Jiang and Yandong Ji and Shoujie Li and Jiangmiao Pang and Houde Liu and Huazhe Xu},
      year={2024},
      eprint={2403.17367},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
}
```

