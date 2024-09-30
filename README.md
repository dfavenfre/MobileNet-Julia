# MobileNet v1 Implementation in Julia - CIFAR100

This repository contains a custom implementation of MobileNet v1, based on the research paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), using the Julia (Flux). The model was trained on the CIFAR100 dataset, and you’ll find both the implementation and results documented here.

## MobileNet v1 Overview
MobileNet v1 is a lightweight and efficient deep learning model designed for mobile and embedded devices. It achieves a balance between performance and computational efficiency by using **depthwise separable convolutions**, which significantly reduce the number of parameters and computational overhead compared to standard convolutions.

## MobileNet v1 Architecture
The core architecture leverages depthwise separable convolutions, as shown in the code below:

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
* Depthwise Separable Convolutions: Standard convolutions are split into two parts:
    * Depthwise Convolutions: Applies a filter to each input channel individually.
    * Pointwise Convolutions: A 1x1 convolution combines the outputs of the depthwise layers. 
    This drastically reduces the number of parameters and computations required for each layer.
* Width Multiplier (α): This allows you to adjust the number of filters (channels) in each layer, enabling you to scale the model for different use cases, from mobile to more compute-heavy environments.

* Resolution Multiplier (ρ): This helps control the resolution of the input image, allowing you to fine-tune the trade-off between speed and accuracy by shrinking or expanding the input size.

MobileNet v1's balance between performance and efficiency makes it ideal for real-time applications on resource-constrained devices, such as smart phones.

# Training
The model was trained on the CIFAR100 dataset, which contains 60,000 images distributed across 100 classes. Training took place over 11 hours on an NVIDIA GeForce RTX 3050 Ti with CUDA enabled and 32 GB of available memory.
The following graphs show the performance of the model during training:

## Validation Accuracy:
![Training and Validation Accuracy](https://github.com/user-attachments/assets/f2d35b0c-5a62-4220-af94-dadf46223cac)

## Training / Validation Loss:
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
