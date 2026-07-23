import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import yaml
import os
from collections import Counter


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cifar10_dataloaders(config):
    """Load CIFAR-10 with Dirichlet Non-IID splits. GPU-ready (pin_memory).
    
    Config keys:
        cifar_root: root directory for CIFAR-10 data (default: ./data, 
                    can be overridden by env var CIFAR10_ROOT or TORCH_HOME)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    use_cuda = torch.cuda.is_available()
    num_workers = config.get("num_workers", 2 if use_cuda else 0)
    pin_memory = config.get("pin_memory", use_cuda)

    # 支持自定义数据根目录：config > env CIFAR10_ROOT > env TORCH_HOME > ./data
    cifar_root = config.get("cifar_root") or os.environ.get("CIFAR10_ROOT") or os.environ.get("TORCH_HOME") or "./data"
    
    try:
        train_set = torchvision.datasets.CIFAR10(
            root=cifar_root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(
            root=cifar_root, train=False, download=True, transform=transform_test)
    except Exception as e:
        print(f"  CIFAR-10 下载失败 ({e})，使用 FakeData 代替")
        train_set = torchvision.datasets.FakeData(
            size=5000, image_size=(3, 32, 32), num_classes=10,
            transform=transform_train)
        test_set = torchvision.datasets.FakeData(
            size=1000, image_size=(3, 32, 32), num_classes=10,
            transform=transform_test)
        train_set.targets = [train_set[i][1] for i in range(len(train_set))]
        test_set.targets = [test_set[i][1] for i in range(len(test_set))]

    num_clients = config["num_clients"]
    alpha = config["dirichlet_alpha"]
    targets = np.array(train_set.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idx_c)).astype(int)
        diff = len(idx_c) - proportions.sum()
        for i in range(diff):
            proportions[i % num_clients] += 1
        start = 0
        for i, p in enumerate(proportions):
            client_indices[i].extend(idx_c[start:start + p].tolist())
            start += p

    train_loaders = []
    for i in range(num_clients):
        subset = Subset(train_set, client_indices[i])
        loader = DataLoader(subset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)
        train_loaders.append(loader)

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loaders, test_loader, client_indices, train_set, test_set


def get_class_subset_loader(dataset, indices, target_class, batch_size=64,
                            num_workers=0, pin_memory=False):
    class_idx = [i for i in indices if dataset.targets[i] == target_class]
    if not class_idx:
        return None
    subset = Subset(dataset, class_idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def apply_label_modification(dataset, indices, modify_ratio, from_class,
                             to_class, seed=42):
    rng = np.random.RandomState(seed)
    from_idx = [i for i in indices if dataset.targets[i] == from_class]
    if not from_idx:
        return []  # no samples of from_class for this client
    n_modify = max(1, int(len(from_idx) * modify_ratio))
    chosen = rng.choice(from_idx, size=n_modify, replace=False).tolist()
    for i in chosen:
        dataset.targets[i] = to_class
    return chosen


def count_class_distribution(dataset, indices):
    return Counter(dataset.targets[i] for i in indices)


def get_client_class_loaders(dataset, client_indices, batch_size=64,
                             num_workers=0, pin_memory=False):
    client_class_loaders = {}
    for cid, indices in enumerate(client_indices):
        class_loaders = {}
        for c in range(10):
            loader = get_class_subset_loader(
                dataset, indices, c, batch_size, num_workers, pin_memory)
            if loader is not None:
                class_loaders[c] = loader
        client_class_loaders[cid] = class_loaders
    return client_class_loaders


def create_clean_loader(dataset, indices, exclude_classes, batch_size=64,
                        num_workers=0, pin_memory=False):
    clean_idx = [i for i in indices if dataset.targets[i] not in exclude_classes]
    if not clean_idx:
        return None
    subset = Subset(dataset, clean_idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_dataset_stats(dataset_name="cifar10"):
    if dataset_name == "cifar10":
        return {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
            "num_classes": 10,
            "input_size": 32,
        }
    elif dataset_name == "cifar100":
        return {
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
            "num_classes": 100,
            "input_size": 32,
        }
    return {"num_classes": 10, "input_size": 32}