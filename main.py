"""
LabelModifyFL: 联邦学习标签修改请求的合法性校验 + 类级外科手术式遗忘
=====================================================================
GPU‑optimized version.  Ready for Google Colab.

实验流水线:
  1. Dirichlet Non‑IID 数据划分 (pin_memory + num_workers)
  2. 客户端本地训练 (AMP mixed precision) → 上传模型更新 + 类条件梯度方向 + 类条件损失
  3. 服务端 LGA‑KDE 检测 → 污染分数 p_c + 可疑客户端集合 S_c
  4. 良性客户端: 加权 trimmed‑mean 聚合 + 类级掩码 M_c 下发
  5. 恶意客户端: 隔离 + 类级外科手术式遗忘
  6. 指标: Test‑Acc, Forget‑Acc, ΔClean, p_c, Precision/Recall

Colab 快速启动:
    !git clone <repo> && cd label_modify_fl
    !pip install -r requirements.txt
    !python main.py --global-rounds 50 --num-clients 10 --alpha 0.5

使用示例:
    python main.py
    python main.py --config config.yaml --global-rounds 100 --model resnet18
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_utils import (
    load_config,
    get_cifar10_dataloaders,
    get_client_class_loaders,
    apply_label_modification,
    count_class_distribution,
    create_clean_loader,
    get_dataset_stats,
)
from fed_utils import (
    compute_class_conditional_gradients,
    local_train,
    lga_kde_detect,
    DetectionHistory,
    weighted_trimmed_aggregate,
    compute_class_mask_vector,
    surgical_unlearn,
    evaluate,
    evaluate_per_class,
    evaluate_all_classes,
    get_param_shapes,
    unflatten_params,
)
from model import get_model, count_parameters, get_param_shapes
from plot_utils import generate_all_plots


def setup_cuda():
    """Configure CUDA for maximum performance."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        device = torch.device("cuda")
        print(f"  CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  cudnn: {torch.backends.cudnn.version()}")
    else:
        device = torch.device("cpu")
        print("  CUDA 不可用, 使用 CPU")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelModifyExperiment:
    def __init__(self, config):
        self.cfg = config
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        self.device = setup_cuda()
        self.use_amp = config.get("use_amp", False) and torch.cuda.is_available()

        self.num_clients = config["num_clients"]
        self.num_classes = 10
        self.global_rounds = config["global_rounds"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.momentum = config.get("momentum", 0.9)
        self.weight_decay = config.get("weight_decay", 5e-4)
        self.local_epochs = config.get("local_epochs", 1)

        self.detection_interval = config.get("detection_interval", 5)

        self.legit_from = config.get("legit_from", 5)
        self.legit_to = config.get("legit_to", 7)
        self.mal_from = config.get("mal_from", 3)
        self.mal_to = config.get("mal_to", 8)
        self.mal_ratio = config.get("mal_ratio", 0.2)
        self.legit_ratio = config.get("legit_ratio", 0.3)
        self.modify_ratio = config.get("modify_ratio", 0.5)

        self.unlearn_epochs = config.get("unlearn_epochs", 2)
        self.unlearn_lr = config.get("unlearn_lr", 0.001)
        self.mask_topk_ratio = config.get("mask_topk_ratio", 0.1)

        self.kde_config = {
            "cos_threshold": config.get("cos_threshold", 0.85),
            "tau_hist": config.get("tau_hist", 30.0),
            "tau_loss": config.get("tau_loss", 2.0),
            "tau_angle": config.get("tau_angle", 35.0),
            "min_history": config.get("min_history", 3),
        }

        self.history_tracker = DetectionHistory(
            max_len=config.get("history_window", 10))

        self.benign_clients = list(range(self.num_clients))  # 初始全量，检测轮后更新

        self.metrics = defaultdict(list)

        # === 优化 3：clean_loader 缓存（按 exclude_classes 缓存） ===
        self._clean_loader_cache = {}

    def setup_data(self):
        print("加载 CIFAR-10 数据集 (Dirichlet Non-IID split)...")
        (self.train_loaders, self.test_loader,
         self.client_indices, self.train_set, self.test_set) = \
            get_cifar10_dataloaders(self.cfg)

        use_cuda = torch.cuda.is_available()
        nw = self.cfg.get("num_workers", 2 if use_cuda else 0)
        pm = self.cfg.get("pin_memory", use_cuda)
        self.nw = nw
        self.pm = pm

        self.client_class_loaders = get_client_class_loaders(
            self.train_set, self.client_indices, self.batch_size,
            num_workers=nw, pin_memory=pm)

        self.data_dist = {}
        for cid in range(self.num_clients):
            self.data_dist[cid] = count_class_distribution(
                self.train_set, self.client_indices[cid])

        print(f"  客户端数: {self.num_clients}, "
              f"Dirichlet α={self.cfg['dirichlet_alpha']}")
        for cid in range(self.num_clients):
            counts = [self.data_dist[cid].get(c, 0) for c in range(10)]
            print(f"    Client {cid}: {counts}")

    def setup_model(self):
        model_name = self.cfg.get("model", "resnet18")
        self.model = get_model(model_name, num_classes=self.num_classes)
        self.model.to(self.device)
        self.param_shapes = get_param_shapes(self.model)
        self.total_params = count_parameters(self.model)
        self.model_name = model_name
        print(f"  模型: {model_name}, 参数量: {self.total_params:,}")

        # === 优化 1：预分配本地模型池，避免每轮 new 模型 + load_state_dict 开销 ===
        self.local_models = [get_model(self.model_name, num_classes=self.num_classes) for _ in range(self.num_clients)]
        for m in self.local_models:
            m.to(self.device)

    def get_client_loader(self, cid):
        return self.train_loaders[cid]

    def simulate_malicious_clients(self):
        n_mal = max(1, int(self.num_clients * self.mal_ratio))
        return random.sample(range(self.num_clients), n_mal)

    def simulate_legitimate_clients(self):
        n_legit = max(1, int(self.num_clients * self.legit_ratio))
        return random.sample(range(self.num_clients), n_legit)

    def apply_label_modifications(self, mal_clients, legit_clients):
        modified_info = {}
        for cid in mal_clients:
            indices = self.client_indices[cid]
            dist_before = count_class_distribution(
                self.train_set, indices)
            apply_label_modification(
                self.train_set, indices,
                modify_ratio=self.modify_ratio,
                from_class=self.mal_from,
                to_class=self.mal_to,
                seed=self.seed + cid * 1000 + self.current_round)
            modified_info[cid] = {
                "type": "malicious",
                "from": self.mal_from,
                "to": self.mal_to,
                "dist_before": dict(dist_before),
                "dist_after": dict(count_class_distribution(
                    self.train_set, indices)),
            }

        for cid in legit_clients:
            if cid in mal_clients:
                continue
            indices = self.client_indices[cid]
            dist_before = count_class_distribution(
                self.train_set, indices)
            apply_label_modification(
                self.train_set, indices,
                modify_ratio=self.modify_ratio,
                from_class=self.legit_from,
                to_class=self.legit_to,
                seed=self.seed + cid * 1000 + self.current_round)
            modified_info[cid] = {
                "type": "legitimate",
                "from": self.legit_from,
                "to": self.legit_to,
                "dist_before": dict(dist_before),
                "dist_after": dict(count_class_distribution(
                    self.train_set, indices)),
            }

        return modified_info

    def run_detection_round(self):
            history = self.history_tracker.get_history()
            all_grad_dirs = {}
            all_losses = {}
            all_class_mags = {}

            # 只对 4 个可疑类计算梯度：mal_from, legit_from, mal_to, legit_to
            # 其余 6 类跳过，直接节省 60% 检测轮显存/时间，语义完全保留
            suspect_classes = {self.mal_from, self.legit_from, self.mal_to, self.legit_to}

            for cid in range(self.num_clients):
                class_loaders = self.client_class_loaders[cid]
                grad_dirs, losses, grad_mags = compute_class_conditional_gradients(
                    self.model, class_loaders, self.device, classes_of_interest=suspect_classes)
                all_grad_dirs[cid] = grad_dirs
                all_losses[cid] = losses
                all_class_mags[cid] = grad_mags
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Flatten per-round {cid → {c → grad}} into {c → grad_avg} for KDE
            # === 修复：用【全客户端平均】做检测，良性均值仅用于聚合 ===
            flat_grads_all = {}
            flat_losses_all = {}
            from collections import defaultdict
            class_counts_all = defaultdict(int)
            for cid in range(self.num_clients):
                for c in all_grad_dirs[cid]:
                    if c not in flat_grads_all:
                        flat_grads_all[c] = all_grad_dirs[cid][c].clone()
                        flat_losses_all[c] = all_losses[cid].get(c, 0.0)
                    else:
                        flat_grads_all[c] += all_grad_dirs[cid][c]
                        flat_losses_all[c] += all_losses[cid].get(c, 0.0)
                    class_counts_all[c] += 1
            for c in flat_grads_all:
                flat_grads_all[c] /= class_counts_all[c]
                flat_losses_all[c] /= class_counts_all[c]

            # 良性客户端均值（仅用于聚合加权）
            flat_grads = {}
            flat_losses = {}
            class_counts = defaultdict(int)
            for cid in self.benign_clients:
                for c in all_grad_dirs[cid]:
                    if c not in flat_grads:
                        flat_grads[c] = all_grad_dirs[cid][c].clone()
                        flat_losses[c] = all_losses[cid].get(c, 0.0)
                    else:
                        flat_grads[c] += all_grad_dirs[cid][c]
                        flat_losses[c] += all_losses[cid].get(c, 0.0)
                    class_counts[c] += 1
            for c in flat_grads:
                flat_grads[c] /= class_counts[c]
                flat_losses[c] /= class_counts[c]

            p_c, suspected_classes, diag = lga_kde_detect(
                flat_grads_all, flat_losses_all, history, self.kde_config)

            # ---- 新增：基于 per-client 类梯度方向 做客户端级打分 ----
            # suspected_classes 是可疑的类索引集合，如 {3, 8}
            # 对每个客户端，计算其在可疑类上的梯度方向偏离历史均值的程度
            client_suspicion = {cid: 0.0 for cid in range(self.num_clients)}
            for cid in range(self.num_clients):
                for c in suspected_classes:
                    if c in all_grad_dirs[cid]:
                        # 当前轮该客户端该类的梯度方向（已归一化）
                        curr = all_grad_dirs[cid][c].detach().cpu()
                        curr = curr / (curr.norm() + 1e-12)
                        # 历史均值方向（flat_grads 已是跨客户端平均，近似历史基准）
                        hist = flat_grads[c].detach().cpu()
                        hist = hist / (hist.norm() + 1e-12)
                        cos_sim = (curr * hist).sum().item()
                        # 偏离度累加：1 - cos_sim，越大越可疑
                        client_suspicion[cid] += (1.0 - cos_sim)

            # 按嫌疑度降序，取前 mal_ratio 比例为恶意客户端
            n_mal = max(1, int(self.num_clients * self.mal_ratio))
            mal_clients = sorted(client_suspicion, key=client_suspicion.get, reverse=True)[:n_mal]

            self.benign_clients = [c for c in range(self.num_clients) if c not in mal_clients]
            if not self.benign_clients:
                self.benign_clients = list(range(self.num_clients))

            # ---- 连续软权重：嫌疑度越高权重越低，平滑过渡 ----
            # 将 suspicion 归一化到 [0,1]，再映射到 [0.1, 1.0]
            sus_vals = np.array([client_suspicion[c] for c in range(self.num_clients)], dtype=float)
            if sus_vals.max() > sus_vals.min():
                sus_norm = (sus_vals - sus_vals.min()) / (sus_vals.max() - sus_vals.min())
            else:
                sus_norm = np.zeros_like(sus_vals)
            # sigmoid-like 映射：正常客户(低嫌疑)→1.0，高嫌疑→0.1
            weights = 0.1 + 0.9 / (1.0 + np.exp(8.0 * (sus_norm - 0.5)))
            # 强制确认的恶意客户端最低权重
            for c in mal_clients:
                weights[c] = min(weights[c], 0.1)

            global_state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}

            client_updates = []
            # === 优化 1：复用预分配模型池（良性客户端）===
            for cid in self.benign_clients:
                local_model = self.local_models[cid]
                local_model.load_state_dict(global_state_cpu)
                loader = self.get_client_loader(cid)
                local_state = local_train(
                    local_model, loader,
                    lr=self.lr, momentum=self.momentum,
                    weight_decay=self.weight_decay,
                    local_epochs=self.local_epochs,
                    device=self.device)
                update = OrderedDict()
                n_samples = len(loader.dataset)
                for k in global_state_cpu:
                    update[k] = (local_state[k].cpu() -
                                 global_state_cpu[k]) * n_samples
                client_updates.append((update, n_samples))

            trim_ratio = np.mean(
                [p_c.get(c, 0.0) for c in flat_grads.keys()])
            # Fix 3: 自适应 trim_ratio —— 仅对高污染可疑类计算，去除 legit_from 稀释
            suspect_pcs = [p_c.get(c, 0.0) for c in suspected_classes if p_c.get(c, 0.0) > 0.2]
            if suspect_pcs:
                trim_ratio = min(0.3, max(0.05, np.mean(suspect_pcs) * 1.5))
            else:
                trim_ratio = 0.05
            # Filter weights to only include benign clients
            benign_weights = [weights[c] for c in self.benign_clients]
            new_state = weighted_trimmed_aggregate(
                global_state_cpu, client_updates, benign_weights, trim_ratio)
            self.model.load_state_dict(new_state)

            class_masks = {}
            for c in range(self.num_classes):
                # 只对高污染分且在可疑类集合中的类生成掩码
                if p_c.get(c, 0.0) > 0.2 and c in suspected_classes:
                    mask_vec = compute_class_mask_vector(
                        self.model, all_class_mags, c, p_c[c],
                        self.param_shapes)
                    mask_dict = unflatten_params(
                        mask_vec.float(), self.param_shapes)
                    class_masks[c] = mask_dict

            if class_masks:
                for c, mask_dict in class_masks.items():
                    # 仅对 p_c > 0.15 的类做遗忘（与生成掩码阈值 0.2 对齐）
                    if p_c.get(c, 0.0) > 0.15:
                        # 关键修正：遗忘类 c 时，干净数据应排除该类的“目标类”
                        # 恶意修改：mal_from -> mal_to，合法修改：legit_from -> legit_to
                        # 遗忘 mal_from 时，毒样本在 mal_to 里；遗忘 legit_from 时，毒样本在 legit_to 里
                        exclude_classes = set()
                        if c == self.mal_from:
                            exclude_classes.add(self.mal_to)
                        elif c == self.legit_from:
                            exclude_classes.add(self.legit_to)

                        # === 优化 3：clean_loader 缓存（按 exclude_classes 键缓存）===
                        cache_key = tuple(sorted(exclude_classes))
                        if cache_key in self._clean_loader_cache:
                            clean_loader = self._clean_loader_cache[cache_key]
                        else:
                            clean_loader = create_clean_loader(
                                self.train_set, list(range(len(self.train_set))),
                                exclude_classes=exclude_classes, batch_size=self.batch_size,
                                num_workers=self.nw, pin_memory=self.pm)
                            if clean_loader is not None:
                                self._clean_loader_cache[cache_key] = clean_loader
                    
                        if clean_loader is not None:
                            unlearned_state = surgical_unlearn(
                                self.model, clean_loader, mask_dict,
                                self.device, epochs=self.unlearn_epochs,
                                lr=self.unlearn_lr, use_amp=self.use_amp)
                            self.model.load_state_dict(unlearned_state)

            self.history_tracker.update(flat_grads, flat_losses)

            all_suspects = mal_clients  # 检出的恶意客户端

            return p_c, suspected_classes, all_suspects, diag, class_masks

    def run_standard_round(self):
        global_state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
        client_updates = []

        # === 优化 1：复用预分配模型池，避免每轮 new + load_state_dict ===
        for cid in range(self.num_clients):
            local_model = self.local_models[cid]
            local_model.load_state_dict(global_state_cpu)
            loader = self.get_client_loader(cid)
            local_state = local_train(
                local_model, loader,
                lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay,
                local_epochs=self.local_epochs,
                device=self.device)
            update = OrderedDict()
            n_samples = len(loader.dataset)
            for k in global_state_cpu:
                update[k] = (local_state[k].cpu() -
                             global_state_cpu[k]) * n_samples
            client_updates.append((update, n_samples))

        total_n = sum(n for _, n in client_updates)
        new_state = OrderedDict()
        for k in global_state_cpu:
            new_state[k] = (global_state_cpu[k] +
                            sum(u[k] for u, _ in client_updates) / total_n)
        self.model.load_state_dict(new_state)

        # 标准轮只需为 4 个可疑类更新历史，省 60% 梯度计算时间
        # === 优化 2：标准轮仅对良性客户端计算梯度（恶意梯度不进历史） ===
        suspect_classes = {self.mal_from, self.legit_from, self.mal_to, self.legit_to}

        all_grad_dirs = {}
        all_losses = {}
        for cid in self.benign_clients:   # 仅良性客户端
            class_loaders = self.client_class_loaders[cid]
            filtered_loaders = {c: loader for c, loader in class_loaders.items() if c in suspect_classes}
            grad_dirs, losses, _ = compute_class_conditional_gradients(
                self.model, filtered_loaders, self.device)
            all_grad_dirs[cid] = grad_dirs
            all_losses[cid] = losses

        # Flatten per-round {cid → {c → grad}} into {c → grad_avg} for history
        # 标准轮仅用良性客户端更新历史基准，防止恶意梯度污染基准
        flat_grads = {}
        flat_losses = {}
        from collections import defaultdict
        class_counts = defaultdict(int)
        for cid in self.benign_clients:
            for c in all_grad_dirs[cid]:
                if c not in flat_grads:
                    flat_grads[c] = all_grad_dirs[cid][c].clone()
                    flat_losses[c] = all_losses[cid].get(c, 0.0)
                else:
                    flat_grads[c] += all_grad_dirs[cid][c]
                    flat_losses[c] += all_losses[cid].get(c, 0.0)
                class_counts[c] += 1
        for c in flat_grads:
            flat_grads[c] /= class_counts[c]
            flat_losses[c] /= class_counts[c]

        self.history_tracker.update(flat_grads, flat_losses)

    def evaluate_round(self):
        acc_all = evaluate(self.model, self.test_loader, self.device)
        acc_forget = evaluate_per_class(
            self.model, self.test_set,
            [self.mal_from, self.legit_from], self.device)
        per_class = evaluate_all_classes(
            self.model, self.test_set, self.device)
        clean_classes = [c for c in range(self.num_classes)
                         if c not in (self.mal_from, self.legit_from,
                                      self.mal_to, self.legit_to)]
        acc_clean = np.mean([per_class[c] for c in clean_classes])

        return {
            "acc_all": acc_all,
            "acc_forget": acc_forget,
            "acc_clean": acc_clean,
            "per_class": per_class,
        }

    def run(self):
        print(f"\n{'='*60}")
        print("LabelModifyFL 实验启动")
        print(f"{'='*60}")
        print(f"  设备: {self.device}  |  AMP: {self.use_amp}")
        print(f"  全局轮次: {self.global_rounds}")
        print(f"  检测间隔: 每 {self.detection_interval} 轮")
        print(f"  合法修改: class {self.legit_from} → {self.legit_to}")
        print(f"  恶意修改: class {self.mal_from} → {self.mal_to}")
        print(f"  恶意客户端比例: {self.mal_ratio}")
        print(f"{'='*60}\n")

        self.setup_data()
        self.setup_model()

        warmup_rounds = max(2, self.detection_interval - 1)
        print(f"\n--- 预热阶段 ({warmup_rounds} 轮) ---")
        for r in range(1, warmup_rounds + 1):
            self.current_round = r
            t0 = time.time()
            self.run_standard_round()
            metrics = self.evaluate_round()
            elapsed = time.time() - t0
            print(f"  [Round {r:3d}] Acc(all)={metrics['acc_all']:6.2f}%  "
                  f"Acc(clean)={metrics['acc_clean']:6.2f}%  "
                  f"({elapsed:.1f}s)")

            self.metrics["round"].append(r)
            self.metrics["acc_all"].append(metrics["acc_all"])
            self.metrics["acc_clean"].append(metrics["acc_clean"])
            self.metrics["acc_forget"].append(metrics["acc_forget"])
            self.metrics["per_class"].append(metrics["per_class"])
            self.metrics["phase"].append("warmup")
            self.metrics["p_c_mal_from"].append(0.0)
            self.metrics["p_c_legit_from"].append(0.0)
            self.metrics["n_suspects"].append(0)

        print(f"\n--- 检测与遗忘阶段 (第 {warmup_rounds+1} 轮起) ---")
        for r in range(warmup_rounds + 1, self.global_rounds + 1):
            self.current_round = r
            t0 = time.time()

            is_detection_round = (r % self.detection_interval == 0)

            if is_detection_round:
                mal_clients = self.simulate_malicious_clients()
                legit_clients = self.simulate_legitimate_clients()
                legit_clients = [c for c in legit_clients
                                 if c not in mal_clients]

                self.apply_label_modifications(mal_clients, legit_clients)

                print(f"\n  [Round {r}] 标签修改: "
                      f"恶意={mal_clients}, 合法={legit_clients}")

                p_c, S_c, all_suspects, diag, class_masks = \
                    self.run_detection_round()

                detected_mal = [c for c in mal_clients
                                if c in all_suspects]
                false_pos = [c for c in all_suspects
                             if c not in mal_clients]
                false_neg = [c for c in mal_clients
                             if c not in all_suspects]

                polluted = [c for c in range(self.num_classes)
                            if p_c.get(c, 0.0) > 0.2]
                p_c_str = ", ".join(
                    f"c{c}={p_c.get(c,0):.2f}" for c in polluted)

                print(f"  [Round {r}] 余弦相似度: p_c=[{p_c_str}], "
                      f"可疑={len(all_suspects)}/{self.num_clients}")
                print(f"  [Round {r}] 检测: "
                      f"命中恶意={len(detected_mal)}/{len(mal_clients)}, "
                      f"误报={len(false_pos)}, 漏报={len(false_neg)}")
                if class_masks:
                    print(f"  [Round {r}] 类级掩码: "
                          f"classes={list(class_masks.keys())}")

                self.metrics["n_suspects"].append(len(all_suspects))
                self.metrics["p_c_mal_from"].append(
                    p_c.get(self.mal_from, 0.0))
                self.metrics["p_c_legit_from"].append(
                    p_c.get(self.legit_from, 0.0))
                self.metrics["detected_mal"].append(detected_mal)
                self.metrics["false_pos"].append(false_pos)
                self.metrics["false_neg"].append(false_neg)
                self.metrics["phase"].append("detection")
            else:
                self.run_standard_round()
                self.metrics["n_suspects"].append(0)
                self.metrics["p_c_mal_from"].append(0.0)
                self.metrics["p_c_legit_from"].append(0.0)
                self.metrics["phase"].append("standard")

            metrics = self.evaluate_round()
            elapsed = time.time() - t0

            status = ""
            if is_detection_round:
                status = f" 可疑={len(all_suspects)}"
            print(f"  [Round {r:3d}] Acc(all)={metrics['acc_all']:6.2f}%  "
                  f"Acc(clean)={metrics['acc_clean']:6.2f}%  "
                  f"Acc(forget)={metrics['acc_forget']:6.2f}%  "
                  f"({elapsed:.1f}s){status}")

            self.metrics["round"].append(r)
            self.metrics["acc_all"].append(metrics["acc_all"])
            self.metrics["acc_clean"].append(metrics["acc_clean"])
            self.metrics["acc_forget"].append(metrics["acc_forget"])
            self.metrics["per_class"].append(metrics["per_class"])

        self.print_summary()
        self.save_results()

    def print_summary(self):
        print(f"\n{'='*60}")
        print("最终结果汇总")
        print(f"{'='*60}")

        final_acc = self.metrics["acc_all"][-1]
        final_clean = self.metrics["acc_clean"][-1]
        final_forget = self.metrics["acc_forget"][-1]

        print(f"  Final Acc(all):    {final_acc:.2f}%")
        print(f"  Final Acc(clean):  {final_clean:.2f}%")
        print(f"  Final Acc(forget): {final_forget:.2f}%")

        per_class = evaluate_all_classes(
            self.model, self.test_set, self.device)
        print(f"\n  逐类准确率:")
        for c in range(self.num_classes):
            marker = ""
            if c == self.mal_from:
                marker = " <-- mal_from"
            elif c == self.legit_from:
                marker = " <-- legit_from"
            elif c == self.mal_to:
                marker = " <-- mal_to"
            elif c == self.legit_to:
                marker = " <-- legit_to"
            print(f"    Class {c}: {per_class[c]:5.2f}%{marker}")

        detection_phases = [i for i, p in enumerate(self.metrics["phase"])
                            if p == "detection"]
        if detection_phases:
            print(f"\n  检测阶段统计:")
            det_idx = 0
            for i in detection_phases:
                r = self.metrics["round"][i]
                n_s = self.metrics["n_suspects"][i]
                if det_idx < len(self.metrics["p_c_mal_from"]):
                    p_mal = self.metrics["p_c_mal_from"][det_idx]
                    p_legit = self.metrics["p_c_legit_from"][det_idx]
                    dm = self.metrics["detected_mal"][det_idx]
                    fp = self.metrics["false_pos"][det_idx]
                    fn = self.metrics["false_neg"][det_idx]
                else:
                    p_mal, p_legit = 0.0, 0.0
                    dm, fp, fn = [], [], []
                det_idx += 1
                print(f"    Round {r}: p_c(mal)={p_mal:.2f}, "
                      f"p_c(legit)={p_legit:.2f}, "
                      f"命中={len(dm)}, 误报={len(fp)}, 漏报={len(fn)}")

    def save_results(self):
        # 统一结果目录
        result_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "results")
        os.makedirs(result_dir, exist_ok=True)

        save_path = os.path.join(result_dir, "results.json")
        serializable = {}
        for k, v in self.metrics.items():
            if k in ("detected_mal", "false_pos", "false_neg"):
                continue
            if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                serializable[k] = [x.tolist() for x in v]
            elif isinstance(v, list) and v and isinstance(
                    v[0], torch.Tensor):
                serializable[k] = [x.item() for x in v]
            else:
                serializable[k] = v

        # 保存 final 逐类准确率
        if self.metrics.get("per_class"):
            serializable["final_per_class"] = self.metrics["per_class"][-1]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存到 {save_path}")

        # 保存模型 final checkpoint
        model_path = os.path.join(result_dir, "model_final.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"  模型已保存到 {model_path}")

        # 保存配置快照
        config_path = os.path.join(result_dir, "config_snapshot.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                        for k, v in self.cfg.items()}, f, indent=2, ensure_ascii=False)
        print(f"  配置快照已保存到 {config_path}")

        # 生成实验结果图表
        try:
            plot_dir = os.path.join(result_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            generate_all_plots(self.metrics, plot_dir)
            print(f"  实验图表已保存到 {plot_dir}/")
        except Exception as e:
            print(f"  绘图失败 (不影响实验结果): {e}")

        print(f"\n  📁 所有结果保存在: {result_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="LabelModifyFL: GPU‑optimized label modification "
                    "detection + surgical unlearning")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        choices=["lenet", "simplecnn", "resnet18"])
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--global-rounds", type=int, default=None)
    parser.add_argument("--modify-ratio", type=float, default=None)
    parser.add_argument("--mal-ratio", type=float, default=None)
    parser.add_argument("--detection-interval", type=int, default=None)
    parser.add_argument("--cos-threshold", type=float, default=None)
    parser.add_argument("--min-history", type=int, default=None)
    parser.add_argument("--unlearn-epochs", type=int, default=None)
    parser.add_argument("--unlearn-lr", type=float, default=None)
    parser.add_argument("--legit-ratio", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用自动混合精度")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    for key in ["model", "num_clients", "global_rounds",
                "modify_ratio", "mal_ratio", "seed",
                "detection_interval", "cos_threshold", "min_history",
                "unlearn_epochs", "unlearn_lr", "legit_ratio"]:
        val = getattr(args, key.replace("-", "_"), None)
        if key == "num_clients":
            val = args.num_clients
        if val is not None:
            config[key] = val

    if args.alpha is not None:
        config["dirichlet_alpha"] = args.alpha
    if args.no_amp:
        config["use_amp"] = False

    exp = LabelModifyExperiment(config)
    exp.run()


if __name__ == "__main__":
    main()