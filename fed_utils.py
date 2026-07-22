import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
import numpy as np
from model import (flatten_params, unflatten_params, get_param_shapes, normalize_gradient_vector, create_mask_from_vector)


def compute_class_conditional_gradients(model, class_loaders, device, classes_of_interest=None):
    """Single-pass: compute both gradient directions and magnitudes per class.

    Args:
        classes_of_interest: Optional set of class indices to compute gradients for.
                             If provided, only these classes are processed (saves ~60% time for 4/10 classes).
    """
    model.eval()
    param_shapes = get_param_shapes(model)
    grad_dirs = {}
    losses = {}
    grad_mags = {}

    # Filter classes if specified
    if classes_of_interest is not None:
        class_loaders = {c: loader for c, loader in class_loaders.items() if c in classes_of_interest}

    for c, loader in class_loaders.items():
        total_grad = None
        total_grad_abs = None
        total_loss = 0.0
        total_samples = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            mask = (target == c)
            if mask.sum() == 0:
                continue
            data_c = data[mask]
            target_c = target[mask]
            model.zero_grad()
            out = model(data_c)
            loss = nn.CrossEntropyLoss()(out, target_c)
            loss.backward()
            grad_vec = flatten_params(OrderedDict((k, p.grad.detach().clone()) for k, p in model.named_parameters() if p.grad is not None))
            grad_abs_vec = flatten_params(OrderedDict((k, p.grad.detach().clone().abs()) for k, p in model.named_parameters() if p.grad is not None))
            n = len(data_c)
            if total_grad is None:
                total_grad = grad_vec * n
                total_grad_abs = grad_abs_vec * n
            else:
                total_grad += grad_vec * n
                total_grad_abs += grad_abs_vec * n
            total_loss += loss.item() * n
            total_samples += n
        if total_samples > 0:
            avg_grad = total_grad / total_samples
            grad_dirs[c] = normalize_gradient_vector(avg_grad)
            grad_mags[c] = total_grad_abs / total_samples
            losses[c] = total_loss / total_samples
        else:
            zero_grad = torch.zeros(sum(s.numel() for s in param_shapes.values()), device=device)
            grad_dirs[c] = zero_grad
            grad_mags[c] = zero_grad
            losses[c] = 0.0
        model.zero_grad()
    return grad_dirs, losses, grad_mags


def compute_class_gradient_magnitude(model, class_loaders, c, device):
    model.eval()
    param_shapes = get_param_shapes(model)
    total_grad_abs = None
    total_samples = 0
    loader = class_loaders.get(c)
    if loader is None:
        return torch.zeros(sum(s.numel() for s in param_shapes.values()), device=device)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        mask = (target == c)
        if mask.sum() == 0:
            continue
        data_c = data[mask]
        target_c = target[mask]
        model.zero_grad()
        out = model(data_c)
        loss = nn.CrossEntropyLoss()(out, target_c)
        loss.backward()
        grad_vec = flatten_params(OrderedDict((k, p.grad.detach().clone().abs()) for k, p in model.named_parameters() if p.grad is not None))
        n = len(data_c)
        if total_grad_abs is None:
            total_grad_abs = grad_vec * n
        else:
            total_grad_abs += grad_vec * n
        total_samples += n
    if total_samples > 0:
        return total_grad_abs / total_samples
    return torch.zeros(sum(s.numel() for s in param_shapes.values()), device=device)


def local_train(model, train_loader, lr, momentum, weight_decay, local_epochs, device):
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class DetectionHistory:
    def __init__(self, max_len=10):
        self.grad_dirs_history = []
        self.losses_history = []
        self.max_len = max_len

    def append(self, grad_dirs, losses):
        # 关键修复：计算完立即搬到 CPU，列表里只存 CPU 张量，释放 GPU 显存
        grad_dirs_cpu = {c: g.detach().cpu() for c, g in grad_dirs.items()}
        self.grad_dirs_history.append(grad_dirs_cpu)
        self.losses_history.append(losses)
        if len(self.grad_dirs_history) > self.max_len:
            self.grad_dirs_history.pop(0)
            self.losses_history.pop(0)

    def update(self, grad_dirs, losses):
        self.append(grad_dirs, losses)

    def get_history(self):
        return self.grad_dirs_history, self.losses_history

    def ready(self):
        return len(self.grad_dirs_history) >= self.max_len


def lga_kde_detect(flat_grad_dirs, flat_losses, history, kde_config, num_classes=10):
    """
    Cosine similarity based detection (replaces KDE).

    flat_grad_dirs: {c: grad_vec} - averaged gradient directions per class for current round
    flat_losses: {c: loss} - averaged losses per class for current round
    history: tuple (grad_history, loss_history) from DetectionHistory.get_history()
    kde_config: dict with cos_threshold, history_window, min_history, tau_loss

    Returns:
        p_c: {class: pollution_score}
        suspected_classes: set of suspicious classes
        diag: diagnostic info
    """
    grad_history, loss_history = history
    cos_threshold = kde_config.get("cos_threshold", 0.85)  # cos < 0.85 => angle > 32°
    min_history = kde_config.get("min_history", 3)
    tau_loss = kde_config.get("tau_loss", 2.0)

    if len(grad_history) < min_history:
        return {}, set(), {}

    p_c = {}
    suspected_classes = set()

    for c in range(num_classes):
        # Collect historical gradient directions for class c
        hist_grads = []
        for h_grads in grad_history:
            if c in h_grads:
                # 双保险：确保在 CPU 上 stack（history 已在 append 时 .cpu()，这里再保一手）
                hist_grads.append(h_grads[c].cpu())
        if len(hist_grads) < 2:
            p_c[c] = 0.0
            continue

        # Compute historical average direction (unit vector)
        hist_avg = torch.stack(hist_grads).mean(0)
        hist_avg = hist_avg / (hist_avg.norm() + 1e-12)

        # Current round direction (already averaged across clients) - move to CPU for detection
        if c not in flat_grad_dirs:
            p_c[c] = 0.0
            continue

        curr_avg = flat_grad_dirs[c].cpu()
        curr_avg = curr_avg / (curr_avg.norm() + 1e-12)

        # Cosine similarity between current and historical average
        cos_sim = (curr_avg * hist_avg).sum() / (curr_avg.norm() * hist_avg.norm() + 1e-12)

        # Pollution score: 1 - cos_sim (0 = identical direction, 1 = opposite)
        p_c[c] = float(1.0 - cos_sim.item())

        # Flag as suspicious if cosine similarity drops below threshold
        if cos_sim.item() < cos_threshold:
            suspected_classes.add(c)

        # Loss trend check (optional additional signal)
        loss_vals = []
        for h_losses in loss_history:
            if c in h_losses:
                loss_vals.append(h_losses[c])
        if len(loss_vals) >= 2:
            loss_trend = loss_vals[-1] - np.mean(loss_vals[:-1])
            if loss_trend > tau_loss:
                p_c[c] = min(p_c[c] + 0.2, 1.0)  # boost score
                suspected_classes.add(c)

    return p_c, suspected_classes, {}


def weighted_trimmed_aggregate(global_state, client_updates, client_weights, trim_ratio=0.2):
    if not client_updates:
        return OrderedDict(global_state)

    updates_only = [u[0] for u in client_updates]
    n_samples = [u[1] for u in client_updates]
    total_n = sum(n_samples)

    keys = updates_only[0].keys()
    aggregated = OrderedDict()
    for k in keys:
        stacked = torch.stack([u[k].float() for u in updates_only])
        w = torch.tensor(client_weights, dtype=torch.float)
        n = len(updates_only)
        trim = max(0, int(n * trim_ratio))

        if trim > 0 and n - 2 * trim > 0:
            delta = torch.zeros_like(stacked[0])
            for idx in range(stacked[0].numel()):
                flat_idx = tuple(torch.unravel_index(torch.tensor(idx), stacked[0].shape))
                vals = stacked[(slice(None),) + flat_idx]
                sorted_vals, sort_indices = torch.sort(vals)
                sorted_w = w[sort_indices]
                trimmed_vals = sorted_vals[trim:n - trim]
                trimmed_w = sorted_w[trim:n - trim]
                trimmed_w = trimmed_w / (trimmed_w.sum() + 1e-12)
                delta[flat_idx] = (trimmed_vals * trimmed_w).sum()
        else:
            w = w / (w.sum() + 1e-12)
            delta = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(0)

        aggregated[k] = global_state[k].float() + delta / total_n
    return aggregated


def compute_client_weights(p_c, suspected_classes, client_ids, min_weight=0.1):
    """Compute client weights based on detection results."""
    weights = []
    for cid in client_ids:
        weight = 1.0
        for c in suspected_classes:
            if p_c.get(c, 0) > 0.5:
                weight *= 0.5
        weights.append(max(weight, min_weight))
    return weights


def compute_class_mask(p_c, suspected_clients, num_classes, threshold=0.5):
    mask = {}
    for c in range(num_classes):
        if p_c.get(c, 0) > threshold and c in suspected_clients:
            mask[c] = True
        else:
            mask[c] = False
    return mask


def compute_class_mask_vector(model, class_mags, class_idx, p_c, param_shapes, topk_ratio=0.1):
    """
    Compute gradient magnitude based mask vector for a specific class.

    Args:
        model: the model
        class_mags: {cid: {c: mag_tensor}} gradient magnitudes per client per class
        class_idx: target class index
        p_c: pollution score for this class
        param_shapes: parameter shapes dict
        topk_ratio: ratio of top parameters to mask
    Returns:
        flat mask vector (1D tensor) that can be passed to unflatten_params
    """
    # Average gradient magnitudes across clients for this class
    mags = []
    for cid in class_mags:
        if class_idx in class_mags[cid]:
            mags.append(class_mags[cid][class_idx])

    if not mags:
        return torch.zeros(sum(s.numel() for s in param_shapes.values()))

    avg_mag = torch.stack(mags).mean(0)  # [total_params]

    # Adjust topk ratio based on pollution score
    adjusted_ratio = topk_ratio * (1 + p_c)

    # Create mask from magnitude vector
    mask_vec = create_mask_from_vector(avg_mag, adjusted_ratio)
    return mask_vec


def surgical_unlearn(model, clean_loader, mask_dict, device, epochs=2, lr=0.001, use_amp=False):
    """Surgical unlearning: gradient ascent on clean data, masked by class_mask."""
    model.train()
    params_to_unlearn = []
    for name, param in model.named_parameters():
        if name in mask_dict:
            params_to_unlearn.append(name)
    if not params_to_unlearn:
        return model.state_dict()

    optimizer = torch.optim.SGD(
        [p for n, p in model.named_parameters() if n in params_to_unlearn],
        lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for _ in range(epochs):
        for data, target in clean_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(data)
                    loss = criterion(out, target)
                scaler.scale(loss).backward()
            else:
                out = model(data)
                loss = criterion(out, target)
                loss.backward()

            for name, param in model.named_parameters():
                if name in mask_dict and param.grad is not None:
                    mask = mask_dict[name].to(param.device)
                    param.grad = param.grad * (-mask)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

    return model.state_dict()


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def evaluate_per_class(model, test_set, classes, device):
    model.eval()
    correct = {c: 0 for c in classes}
    total = {c: 0 for c in classes}
    from torch.utils.data import DataLoader
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for c in classes:
                mask = (target == c)
                correct[c] += pred[mask].eq(target[mask]).sum().item()
                total[c] += mask.sum().item()
    acc = []
    for c in classes:
        if total[c] > 0:
            acc.append(100.0 * correct[c] / total[c])
        else:
            acc.append(0.0)
    return np.mean(acc)


def evaluate_all_classes(model, test_set, device):
    model.eval()
    correct = [0] * 10
    total = [0] * 10
    from torch.utils.data import DataLoader
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for c in range(10):
                mask = (target == c)
                correct[c] += pred[mask].eq(target[mask]).sum().item()
                total[c] += mask.sum().item()
    per_class = {}
    for c in range(10):
        per_class[c] = 100.0 * correct[c] / total[c] if total[c] > 0 else 0.0
    return per_class