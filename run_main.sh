# main evaluation script for timekeeping experiments
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset CIFAR10 --exp_mode timekeeping --ipc 1 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset CIFAR10 --exp_mode timekeeping --ipc 10 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset CIFAR10 --exp_mode timekeeping --ipc 50 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset MNIST --exp_mode timekeeping --ipc 1 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset MNIST --exp_mode timekeeping --ipc 10 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset MNIST --exp_mode timekeeping --ipc 50 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset FashionMNIST --exp_mode timekeeping --ipc 1 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset FashionMNIST --exp_mode timekeeping --ipc 10 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset FashionMNIST --exp_mode timekeeping --ipc 50 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset SVHN --exp_mode timekeeping --ipc 1 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset SVHN --exp_mode timekeeping --ipc 10 --loss W1
CUDA_VISIBLE_DEVICES=1 /home/limuyang/miniconda3/envs/torchEnv/bin/python main.py --dataset SVHN --exp_mode timekeeping --ipc 50 --loss W1
