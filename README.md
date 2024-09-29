# MobileNet v1 Implementation in Julia - CIFAR100

This repository contains a custom implementation of MobileNet v1 from the research paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) using Julia. The model was trained on the CIFAR100 dataset, and this repository includes both the implementation and the results of the training.

# MobileNet v1 Overview
MobileNet v1 is a class of efficient deep learning models designed for mobile and embedded vision applications. It leverages depthwise separable convolutions, which significantly reduce the number of parameters and computational cost, making the model suitable for devices with limited resources.

## MobileNet v1 Architecture
![image](https://github.com/user-attachments/assets/f5fa61c6-c273-4d83-bea7-68e6711ccecf)

```Julia
function conv_dw(kernel, inp, oup, stride; ρ=0.9997)
    hidden_dim = max(1, Int(round(inp * ρ)))

    return Chain(
        DepthwiseConv((kernel, kernel), inp => inp, stride=stride, pad=1, bias=false, init=Flux.glorot_uniform),
        BatchNorm(inp),
        relu6,
        Conv((1, 1), inp => hidden_dim, stride=1, pad=0, bias=false, init=Flux.glorot_uniform),
        BatchNorm(hidden_dim),
        relu6,
        Conv((1, 1), hidden_dim => oup, stride=1, pad=0, bias=false, init=Flux.glorot_uniform),
        BatchNorm(oup),
        relu6
    )
end
```

![Ekran görüntüsü 2024-09-29 112812](https://github.com/user-attachments/assets/b42696a3-4be7-4107-ae4d-938f14bb2ef6)
```Julia
function MobileNet_V1(n_class; α=0.1)

    function adjust_filters(f)
        return Int(round(α * f))
    end

    return Chain(
        Conv((3, 3), 3 => adjust_filters(32), stride=2, pad=1, bias=false, init=Flux.glorot_uniform),
        BatchNorm(adjust_filters(32)),
        relu6,
        conv_dw(3, adjust_filters(32), adjust_filters(64), 1),
        conv_dw(3, adjust_filters(64), adjust_filters(128), 2),
        conv_dw(3, adjust_filters(128), adjust_filters(128), 1),
        conv_dw(3, adjust_filters(128), adjust_filters(256), 2),
        conv_dw(3, adjust_filters(256), adjust_filters(256), 1),
        conv_dw(3, adjust_filters(256), adjust_filters(512), 2),
        conv_dw(3, adjust_filters(512), adjust_filters(512), 1),
        conv_dw(3, adjust_filters(512), adjust_filters(512), 1),
        conv_dw(3, adjust_filters(512), adjust_filters(512), 1),
        conv_dw(3, adjust_filters(512), adjust_filters(512), 1),
        conv_dw(3, adjust_filters(512), adjust_filters(1024), 2),
        conv_dw(3, adjust_filters(1024), adjust_filters(1024), 1),
        GlobalMeanPool(),
        Flux.flatten,
        Dense(adjust_filters(1024), n_class),
        softmax
    )
end
```
### Key Features of MobileNet v1:
1. **Depthwise Separable Convolutions**: The core idea is to decompose a standard convolution into two simpler operations:
   - **Depthwise Convolution**: Each input channel is convolved independently.
   - **Pointwise Convolution**: A 1x1 convolution is applied to combine the depthwise-convoluted outputs.
   
   This reduces both the computational complexity and the number of parameters compared to traditional convolutions.
   
2. **Width Multiplier (α)**: This hyperparameter reduces the number of channels in each layer by multiplying by a constant, further reducing the size of the model.

3. **Resolution Multiplier (ρ)**: This factor allows controlling the input image resolution, reducing computational complexity by shrinking the spatial resolution of the input images.

Overall, MobileNet v1 strikes a balance between accuracy and efficiency, making it a powerful model for real-time applications on devices with limited computing power.

# Training

MobileNet v1 was trained for 11 uninterrupted hours on an NVIDIA GeForce RTX 3050 Ti CUDA-enabled GPU with 32 GB of available memory.

## Training / Validation Loss:
![Training and Validation Accuracy](https://github.com/user-attachments/assets/f2d35b0c-5a62-4220-af94-dadf46223cac)

## Validation Accuracy:
![Training and Validation Loss](https://github.com/user-attachments/assets/c0ce788c-7b1e-405c-ad62-e22cde2d5a6b)


# Usage
To clone the repository and install the dependencies, you can use the following commands:

```bash
git clone https://github.com/dfavenfre/MobileNet-Julia.git
```

Install required Julia packages:

```Julia
using Pkg
Pkg.instantiate()
```
