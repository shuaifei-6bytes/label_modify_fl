import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================
#  Models
# ============================================================

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def get_model(name="resnet18", num_classes=10):
    if name.lower() == "lenet":
        return LeNet(num_classes=num_classes)
    elif name.lower() == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif name.lower() == "resnet18":
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================
#  Parameter utilities (device‑aware)
# ============================================================

def get_param_shapes(model):
    return OrderedDict((k, p.shape) for k, p in model.named_parameters())


def flatten_params(state_dict):
    vecs = []
    for k in sorted(state_dict.keys()):
        vecs.append(state_dict[k].flatten())
    return torch.cat(vecs)


def unflatten_params(flat_vec, param_shapes):
    state = OrderedDict()
    offset = 0
    device = flat_vec.device
    for k in sorted(param_shapes.keys()):
        shape = param_shapes[k]
        n = shape.numel()
        state[k] = flat_vec[offset:offset + n].view(shape).to(device)
        offset += n
    return state


def normalize_gradient_vector(grad_vec):
    norm = grad_vec.norm()
    if norm < 1e-12:
        return torch.zeros_like(grad_vec)
    return grad_vec / norm


def create_mask_from_vector(grad_vec, ratio):
    """
    Soft mask using sigmoid scaling based on gradient magnitude percentile.
    Returns continuous values in [0, 1] instead of hard 0/1.
    """
    k = max(1, int(len(grad_vec) * ratio))
    _, top_indices = torch.topk(grad_vec.abs(), k)
    threshold = grad_vec.abs()[top_indices].min()  # Top-K 最小幅度作为阈值
    
    # Soft mask: sigmoid(scale * (|grad| - threshold))
    # scale 控制过渡陡峭度，越大越接近硬掩码
    scale = 50.0
    mask = torch.sigmoid(scale * (grad_vec.abs() - threshold))
    return mask


def mask_vector_to_dict(mask_vector, param_shapes):
    return unflatten_params(mask_vector.float(), param_shapes)


def apply_mask_to_gradients(model, mask_dict):
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask_dict:
            m = mask_dict[name].to(param.device)
            param.grad = param.grad * (1 - m)


def freeze_masked_params(model, mask_dict):
    for name, param in model.named_parameters():
        if name in mask_dict:
            mask_flat = mask_dict[name].flatten()
            if mask_flat.sum() > 0:
                param.requires_grad_(False)


def unfreeze_all_params(model):
    for param in model.parameters():
        param.requires_grad_(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)