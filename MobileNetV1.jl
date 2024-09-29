using Flux, MLDatasets, NNlib
using CUDA, cuDNN
using Flux: onehotbatch, onecold, Optimiser, DepthwiseConv, BatchNorm, GlobalMeanPool
using ProgressBars
using MLUtils: DataLoader
using StatsBase
using NNlib
using Wandb, Dates, Logging
using Parameters
using Random
using Augmentor
include("tools.jl")

@kwdef struct training_args
    Wandb_Name::String = "MOBILENETv1"
    project_name::String = "TOLGA_ŞAKAR"
    η::Float64 = 3e-4
    use_cuda::Bool = true
    CudaDevice::Int = 0
    n_epochs::Int = 32
    num_classes::Int = 100
    seed::Int = 0
    batch_size::Int = 1
end


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


function get_statistics(dataset::DataType)
    data_set = dataset(:train)[:][1]
    return mean(data_set, dims=[1, 2, 4]), std(data_set, dims=[1, 2, 4])
end


function get_data(batchsize::Int64)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    xtrain, x_train_coarse, ytrain = CIFAR100.traindata()
    xtest, x_train_coarse, ytest = CIFAR100.testdata()

    # Normalize -- these dudes may be recalculated for each run--
    m_xtrain, m_xtest = mean(xtrain, dims=[1, 2, 4]), mean(xtest, dims=[1, 2, 4])
    s_xtrain, s_xtest = std(xtrain, dims=[1, 2, 4]), std(xtest, dims=[1, 2, 4])

    xtrain = @. (xtrain - m_xtrain) / s_xtrain
    xtest = @. (xtest - m_xtest) / s_xtest

    ytrain, ytest = onehotbatch(ytrain, 0:99), onehotbatch(ytest, 0:99)

    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true, parallel=true, buffer=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize, parallel=true, buffer=true)
    @info "Dataset preprocessing is done!!!"
    return train_loader, test_loader
end

function cosine_annealing_lr(η_max, η_min, epoch)
    if epoch <= 8
        return η_min + 0.5 * (η_max - η_min) * (1 + cos(pi * epoch / 8))
    else
        return η_min
    end
end


function train(args::training_args)

    ## Extract params from args
    η = args.η
    use_cuda = args.use_cuda
    cuda_device = args.CudaDevice
    num_classes = args.num_classes
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    project_name = args.project_name
    Wandb_Name = args.Wandb_Name


    train_loader, test_loader = get_data(batch_size)

    if use_cuda
        device = gpu
        CUDA.device!(cuda_device)
        @info "Training on GPU:$cuda_device"

    else
        device = cpu
        @info "Training on CPU"
    end

    model = begin
        MobileNet_V1(num_classes) |> device
    end

    lg = WandbLogger(project=project_name, name=Wandb_Name * "-$(now())", config=Dict("architecture" => "CNN", "dataset" => "CIFAR-100"))
    global_logger(lg)
    train_loss = loss_reg()
    val_loss = loss_reg()

    for epoch in 1:n_epochs

        updated_η = cosine_annealing_lr(η / 10, η * 10, n_epochs)
        opt = Optimiser(ADAM(updated_η))
        opt_state = Flux.setup(opt, model)

        for (x, y) in ProgressBar(train_loader)
            x, y = map(device, [x, y])
            y = Flux.label_smoothing(y, 0.1f0)
            loss, grads = Flux.withgradient(model) do model
                Flux.logitcrossentropy(model(x), y)
            end
            update!(train_loss, loss |> cpu)
            Flux.update!(opt_state, model, grads[1])
        end

        acc = 0.0f0
        m = 0
        for (x, y) in test_loader
            x, y = map(device, [x, y])
            z = model(x)
            temp_validation_loss = Flux.logitcrossentropy(model(x), y)
            update!(val_loss, temp_validation_loss |> cpu)
            acc += sum(onecold(z) .== onecold(y)) |> cpu
            m += size(x)[end]
        end

        Wandb.log(lg, Dict("Training loss" => get_avg(train_loss), "Validation Accuracy" => acc / m, "Validation Loss" => get_avg(val_loss), "Learning Rate" => updated_η, "Epoch" => epoch, "Batch Size" => batch_size))
        map(reset!, [train_loss, val_loss])
    end
    close(lg)
end

args = training_args()
train(args)

