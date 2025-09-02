# diff_envshade: Differentiable Shader for Environment Lighting in PyTorch

`diff_envshade` is a **differentiable, CUDA-accelerated PyTorch extension** for shading 3D models under **environment map lighting**.

- **Input:** per-pixel surface attributes (e.g., normals, BRDF material parameters) and an environment map (lat-long rectangular format)  
- **Output:** per-pixel RGB colors  

Shading is computed via brute-force integration: each texel in the environment map is treated as a directional light, and contributions are accumulated according to the BRDF. This simple yet effective approach ensures fully differentiable gradients, is well-suited for non-highly-specular materials (e.g., human faces), and a 16×32 environment map resolution provides a good balance between performance and quality.

This shader can even account for omnidirectional visibility, taking occlusions from all lighting directions in the environment map into consideration. The occlusions themselves can be efficiently computed using [`torchoptixext_visibility`](https://github.com/zjumsj/torchoptixext_visibility). With visibility considered, shading 70k 3D Gaussians can still be completed **within 1 ms** (tested on an RTX 4090).

## Installation  
Make sure you have a Python environment with PyTorch installed, then run:  
```shell
git clone https://github.com/zjumsj/diff_envshade
cd diff_envshade
python setup.py install
```

## Cite
Please kindly cite our repository and preceding paper if you find our software or algorithm useful for your research.
```
@misc{ma2025diffenvshade,
    title={Differentiable Shader for Environment Lighting in PyTorch},
    author={Ma, Shengjie},
    year={2025},
    month={aug},
    url={https://github.com/zjumsj/diff_envshade}
}
```
```
@inproceedings{ma2025relightable,
    title={Relightable Gaussian Blendshapes for Head Avatar Animation},
    author={Ma, Shengjie and Zheng, Youyi and Weng, Yanlin and Zhou, Kun},
    booktitle={International Conference on Computer-Aided Design and Compute Graphics},
    year={2025}
}
```



