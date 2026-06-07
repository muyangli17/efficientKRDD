import os
import random
import copy
import argparse
import time
import queue
import numpy as np
import torch
from torch.optim import Adam, SGD, RMSprop
from torchvision.utils import save_image
from tqdm import tqdm
from utils import (
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    DiffAugment,
    ParamDiffAug,
    get_time,
)
from calculateWassersteindistance import calculateW1
from criticfun import CriticFun


def main():
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset")
    parser.add_argument(
        "--feature_extractor", type=str, default="ConvNet", help="model"
    )
    parser.add_argument("--evaluator", type=str, default="ConvNet", help="model")
    parser.add_argument("--ipc", type=int, default=10, help="image(s) per class")
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="detailed",
        help="experiment mode(detailed/timekeeping)",
    )  # time
    parser.add_argument("--num_exp", type=int, default=1, help="number of experiments")
    parser.add_argument(
        "--num_eval",
        type=int,
        default=5,
        help="number of evaluating randomly initialized models",
    )
    parser.add_argument(
        "--epoch_eval_train",
        type=int,
        default=1000,
        help="epoch interval to train a model with synthetic data",
    )  # it can be small for speeding up with little performance drop
    parser.add_argument(
        "--Iteration", type=int, default=20000, help="training iterations"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=2000, help="evaluation frequency"
    )

    parser.add_argument(
        "--lr_img",
        type=float,
        default=1.0,
        help="learning rate for updating synthetic images",
    )
    parser.add_argument(
        "--lr_net",
        type=float,
        default=0.01,
        help="learning rate for updating network parameters",
    )

    parser.add_argument(
        "--batch_real", type=int, default=256, help="batch size for real data"
    )
    parser.add_argument(
        "--batch_train", type=int, default=256, help="batch size for training networks"
    )

    parser.add_argument(
        "--init",
        type=str,
        default="real",
        help="noise/real: initialize synthetic images from random noise or randomly sampled real images.",
    )
    parser.add_argument(
        "--dsa_strategy",
        type=str,
        default="color_crop_cutout_flip_scale_rotate",
        help="differentiable Siamese augmentation strategy",
    )

    parser.add_argument("--data_path", type=str, default="../data", help="dataset path")

    parser.add_argument(
        "--loss", type=str, default="MSE", help="loss function (MSE/W1)"
    )

    parser.add_argument(
        "--lip_strategy", type=str, default="SpectralNorm", help="lip_strategy"
    )

    parser.add_argument(
        "--KR_opti",
        type=str,
        default="RMSProp",
        help="optimizor for KR dual (SGD/RMSProp/Adam)",
    )
    parser.add_argument(
        "--warmup_it",
        type=int,
        default=20,
        help="warm up iterations for training critic function",
    )
    parser.add_argument(
        "--critic_it",
        type=int,
        default=5,
        help="number of iterations to optimize KR dual when updating synthetic data",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=1e-3,
        help="learning rate for optimizing KR dual",
    )

    parser.add_argument(
        "--ipfe",
        type=int,
        default=5,
        help="iteration per feature extractor",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    # args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ["none", "None"] else True

    args.exp_name = f"{args.dataset}_{args.feature_extractor}_{args.evaluator}_ipc{args.ipc}_{args.lip_strategy}_{args.init}_{args.loss}_{args.exp_mode}_seed{args.seed}@{get_time()}"

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(args.seed)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists("./results"):
        os.mkdir("./results")

    if not os.path.exists(os.path.join("results", args.exp_name)):
        os.mkdir(os.path.join("results", args.exp_name))

    import logging
    import logging.handlers

    # 配置异步日志：QueueHandler + QueueListener，避免磁盘 I/O 阻塞训练循环
    log_path = os.path.join("results", args.exp_name, args.exp_name + ".log")
    log_queue = queue.Queue(-1)  # 无界队列，主线程写入永不阻塞

    queue_handler = logging.handlers.QueueHandler(log_queue)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # 清除已有 handler，避免重复
    root_logger.addHandler(queue_handler)
    listener.start()

    logging.info(f"logger has been set up (async QueueHandler mode)")

    if args.exp_mode == "detailed":
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_freq).tolist()
    elif args.exp_mode == "timekeeping":
        eval_it_pool = []  # do not evaluate

    logging.info(f"eval_it_pool: {eval_it_pool}")
    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
    ) = get_dataset(args.dataset, args.data_path)
    model_eval_pool = [args.evaluator]

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    exp_time = []
    for exp in range(args.num_exp):
        logging.info(f"================ Exp {exp} ================")
        logging.info(f"Hyper-parameters: \n{args.__dict__}")
        logging.info(f"Evaluation model pool: {model_eval_pool}")

        """ organize the real dataset """
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [
            torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))
        ]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            logging.info(f"class c = {c}: {len(indices_class[c])} real images")

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            logging.info(
                f"real images channel {ch}, mean = {torch.mean(images_all[:, ch]):.4f}, std = {torch.std(images_all[:, ch]):.4f}"
            )

        """ initialize the synthetic data """
        image_syn = torch.randn(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
        # label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        label_syn = torch.tensor(
            np.array([np.ones(args.ipc) * i for i in range(num_classes)]),
            dtype=torch.long,
            requires_grad=False,
            device=args.device,
        ).view(
            -1
        )  # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == "real":
            logging.info("initialize synthetic data from random real images")
            for c in range(num_classes):
                image_syn.data[c * args.ipc : (c + 1) * args.ipc] = (
                    get_images(c, args.ipc).detach().data
                )
        else:
            logging.info("initialize synthetic data from random noise")

        """ training """
        optimizer_img = torch.optim.SGD(
            [image_syn],
            lr=args.lr_img,
            momentum=0.5,
        )  # optimizer_img for synthetic data
        optimizer_img.zero_grad()

        logging.info("training begins")
        t_start = time.time()
        for it in tqdm(range(args.Iteration + 1), desc="updating synthetic data"):

            """Evaluate synthetic data"""
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    logging.info(
                        f"-------------------------\nEvaluation\nfeature_extractor = {args.feature_extractor}, evaluator = {model_eval}, iteration = {it}"
                    )

                    logging.info(f"DSA augmentation strategy: {args.dsa_strategy}")
                    logging.info(
                        f"DSA augmentation parameters: \n{args.dsa_param.__dict__}"
                    )

                    accs = []
                    for it_eval in tqdm(
                        range(args.num_eval),
                        desc=f"evaluate synthetic data at epoch {it}",
                    ):
                        net_eval = get_network(
                            model_eval, channel, num_classes, im_size
                        ).to(
                            args.device
                        )  # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(
                            image_syn.detach()
                        ), copy.deepcopy(
                            label_syn.detach()
                        )  # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(
                            it_eval,
                            net_eval,
                            image_syn_eval,
                            label_syn_eval,
                            testloader,
                            args,
                            logging,
                        )
                        accs.append(acc_test)
                    logging.info(
                        f"Evaluate {len(accs)} random {model_eval}, mean = {np.mean(accs):.4f} std = {np.std(accs):.4f}\n-------------------------"
                    )

                """ visualize and save """
                save_name = os.path.join(
                    "results",
                    args.exp_name,
                    "vis_exp%d_iter%d.png" % (exp, it),
                )
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(
                    image_syn_vis, save_name, nrow=args.ipc
                )  # Trying normalize = True/False may get better visual effects.

            """ Train synthetic data """
            if it % args.ipfe == 0:
                net = get_network(
                    args.feature_extractor, channel, num_classes, im_size
                ).to(
                    args.device
                )  # get a random model
                net.train()

                for param in list(net.parameters()):
                    param.requires_grad = False
                embed = (
                    net.module.embed if torch.cuda.device_count() > 1 else net.embed
                )  # for GPU parallel

                embed_dim = {
                    "ConvNet": 2048,
                    "ResNet18": 512,
                    "VGG11": 512,
                    "AlexNet": 3072,
                }[args.feature_extractor]
                critic_fun_list = [
                    CriticFun(input_dim=embed_dim, SN=True).to(args.device)
                    for _ in range(num_classes)
                ]
                if args.KR_opti == "Adam":
                    opti_KR_list = [
                        Adam(critic_fun_list[c].parameters(), lr=args.critic_lr)
                        for c in range(num_classes)
                    ]
                elif args.KR_opti == "SGD":
                    opti_KR_list = [
                        SGD(
                            critic_fun_list[c].parameters(),
                            lr=args.critic_lr,
                            momentum=0.9,
                        )
                        for c in range(num_classes)
                    ]
                elif args.KR_opti == "RMSProp":
                    opti_KR_list = [
                        RMSprop(critic_fun_list[c].parameters(), lr=args.critic_lr)
                        for c in range(num_classes)
                    ]
                else:
                    raise ValueError("Unknown KR_opti option!")

            loss_avg = 0

            """ update synthetic data """
            # === Phase 1: 收集所有类别的图像并做 DSA（DSA 按类别配对外，轻量操作） ===
            real_list, syn_list = [], []
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c * args.ipc : (c + 1) * args.ipc].reshape(
                    (args.ipc, channel, im_size[0], im_size[1])
                )
                if args.dsa:
                    seed = random.randint(0, 99999)
                    img_real = DiffAugment(
                        img_real, args.dsa_strategy, seed=seed, param=args.dsa_param
                    )
                    img_syn = DiffAugment(
                        img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param
                    )
                real_list.append(img_real)
                syn_list.append(img_syn)

            # === Phase 2: 合并所有类别 → 一次大 batch 前向传播（GPU 利用率的关键优化） ===
            all_real = torch.cat(real_list, dim=0)  # [C * batch_real, ch, h, w]
            all_syn = torch.cat(syn_list, dim=0)  # [C * ipc, ch, h, w]
            all_output_real = embed(all_real).detach()  # 单次大 batch embed
            all_output_syn = embed(all_syn)  # 单次大 batch embed

            # === Phase 3: 按类别切分 embedding 并计算 loss（纯数学运算，无 GPU 调度开销） ===
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                output_real = all_output_real[
                    c * args.batch_real : (c + 1) * args.batch_real
                ]
                output_syn = all_output_syn[c * args.ipc : (c + 1) * args.ipc]

                if args.loss == "MSE":
                    loss += torch.sum(
                        (torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))
                        ** 2
                    )
                elif args.loss == "W1":
                    # 若更新了特征提取器，则重新warm up critic function
                    if it % args.ipfe == 0:
                        p = output_real.detach()
                        q = output_syn.detach()
                        for _ in range(args.warmup_it):
                            opti_KR_list[c].zero_grad()
                            warmup_loss = -calculateW1(p, q, critic_fun_list[c])
                            warmup_loss.backward()
                            opti_KR_list[c].step()
                            if args.lip_strategy == "truncate":
                                critic_fun_list[c].truncate_parameters()
                            elif args.lip_strategy == "SpectralNorm":
                                pass
                    else:
                        for _ in range(args.critic_it):
                            opti_KR_list[c].zero_grad()
                            KR_loss = -calculateW1(p, q, critic_fun_list[c])
                            KR_loss.backward()
                            opti_KR_list[c].step()
                            if args.lip_strategy == "truncate":
                                critic_fun_list[c].truncate_parameters()
                            elif args.lip_strategy == "SpectralNorm":
                                pass

                    loss += calculateW1(output_real, output_syn, critic_fun_list[c])

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            loss_avg = loss.item() / num_classes

            if it % 10 == 0:
                logging.info(f"iter = {it:05d}, loss = {loss_avg:.4f}")

        t_end = time.time()
        exp_time.append(t_end - t_start)

        """Evaluate synthetic data at the end of each experiment"""
        for model_eval in model_eval_pool:
            logging.info(
                f"-------------------------\nEvaluation\nfeature_extractor = {args.feature_extractor}, evaluator = {model_eval}, iteration = {it}"
            )

            logging.info(f"DSA augmentation strategy: {args.dsa_strategy}")
            logging.info(f"DSA augmentation parameters: \n{args.dsa_param.__dict__}")

            accs = []
            for it_eval in tqdm(
                range(args.num_eval),
                desc=f"evaluate synthetic data at epoch {it}",
            ):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                    args.device
                )  # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(
                    image_syn.detach()
                ), copy.deepcopy(
                    label_syn.detach()
                )  # avoid any unaware modification
                _, acc_train, acc_test = evaluate_synset(
                    it_eval,
                    net_eval,
                    image_syn_eval,
                    label_syn_eval,
                    testloader,
                    args,
                    logging,
                )
                accs.append(acc_test)
            logging.info(
                f"Evaluate {len(accs)} random {model_eval}, mean = {np.mean(accs):.4f} std = {np.std(accs):.4f}\n-------------------------"
            )

            accs_all_exps[model_eval] += accs  # record the final results

    logging.info("================ Final Results ================")
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logging.info(
            f"Run {args.num_exp} experiments, feature_extractor = {args.feature_extractor}, evaluate {len(accs)} random {key}, mean  = {np.mean(accs) * 100:.2f}%  std = {np.std(accs) * 100:.2f}%, total time = {np.mean(exp_time)} seconds\n"
        )

    listener.stop()  # 停止后台日志线程，确保日志全部写入磁盘


if __name__ == "__main__":
    main()
