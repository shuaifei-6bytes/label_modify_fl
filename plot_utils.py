import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# 中文字体支持（可选，Colab 通常有 DejaVu Sans）
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def setup_plot_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('ggplot')
        except Exception:
            pass
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })


def plot_training_curves(metrics, save_path, title="Training Curves"):
    """绘制训练曲线：Acc(all), Acc(clean), Acc(forget)"""
    setup_plot_style()
    rounds = metrics.get('round', [])
    if not rounds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 左图：整体准确率
    ax1 = axes[0]
    ax1.plot(rounds, metrics['acc_all'], 'b-o', label='Acc (All)', linewidth=2, markersize=5)
    ax1.plot(rounds, metrics['acc_clean'], 'g-s', label='Acc (Clean)', linewidth=2, markersize=5)
    ax1.plot(rounds, metrics['acc_forget'], 'r-^', label='Acc (Forget)', linewidth=2, markersize=5)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：疑似客户端数量 + p_c
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(rounds, metrics.get('n_suspects', [0]*len(rounds)), 
            alpha=0.5, color='orange', label='Suspects', width=0.8)
    ax2_twin.plot(rounds, metrics.get('p_c_mal_from', [0]*len(rounds)), 
                  'r-o', label='p_c (mal_from)', markersize=4)
    ax2_twin.plot(rounds, metrics.get('p_c_legit_from', [0]*len(rounds)), 
                  'g-s', label='p_c (legit_from)', markersize=4)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('# Suspects', color='orange')
    ax2_twin.set_ylabel('p_c', color='red')
    ax2.set_title('Detection Statistics')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线已保存: {save_path}")


def plot_per_class_heatmap(per_class_history, save_path, title="Per-Class Accuracy Heatmap"):
    """绘制逐类准确率热力图"""
    if not per_class_history:
        return

    setup_plot_style()
    rounds = len(per_class_history)
    classes = 10
    data = np.zeros((rounds, classes))
    for r, pc in enumerate(per_class_history):
        for c in range(classes):
            data[r, c] = pc.get(c, 0.0)

    fig, ax = plt.subplots(figsize=(10, max(6, rounds * 0.4)))
    im = ax.imshow(data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100,
                   interpolation='nearest')
    
    ax.set_yticks(range(classes))
    ax.set_yticklabels([f'Class {i}' for i in range(classes)])
    ax.set_xticks(range(rounds))
    ax.set_xticklabels([f'R{i+1}' for i in range(rounds)])
    ax.set_xlabel('Round')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(rounds):
        for j in range(classes):
            val = data[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(i, j, f'{val:.1f}', ha='center', va='center', 
                    fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  逐类热力图已保存: {save_path}")


def plot_detection_confusion(detected_mal_list, false_pos_list, false_neg_list, 
                              save_path, title="Detection Confusion"):
    """绘制检测混淆矩阵随轮次变化"""
    if not detected_mal_list:
        return

    setup_plot_style()
    rounds = len(detected_mal_list)
    tp = [len(d) for d in detected_mal_list]
    fp = [len(f) for f in false_pos_list]
    fn = [len(f) for f in false_neg_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(rounds)
    width = 0.25
    ax.bar(x - width, tp, width, label='True Positive (命中恶意)', color='green', alpha=0.7)
    ax.bar(x, fp, width, label='False Positive (误报)', color='orange', alpha=0.7)
    ax.bar(x + width, fn, width, label='False Negative (漏报)', color='red', alpha=0.7)
    
    ax.set_xlabel('Detection Round')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'R{i+1}' for i in range(rounds)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  检测混淆图已保存: {save_path}")


def plot_p_c_evolution(p_c_mal_from, p_c_legit_from, save_path, title="p_c Evolution"):
    """绘制 p_c 随轮次演化"""
    if not p_c_mal_from:
        return

    setup_plot_style()
    rounds = len(p_c_mal_from)
    x = np.arange(rounds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, p_c_mal_from, 'r-o', label=f'p_c (mal_from)', linewidth=2, markersize=5)
    ax.plot(x, p_c_legit_from, 'g-s', label=f'p_c (legit_from)', linewidth=2, markersize=5)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.2)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='High Risk (0.5)')
    
    ax.set_xlabel('Detection Round')
    ax.set_ylabel('p_c (Pollution Score)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  p_c 演化图已保存: {save_path}")


def plot_detection_phase_comparison(metrics, save_path, title="Phase Comparison"):
    """对比 Warmup / Standard / Detection 阶段的性能"""
    phases = metrics.get('phase', [])
    if not phases:
        return

    setup_plot_style()
    rounds = len(phases)
    acc_all = metrics.get('acc_all', [])
    acc_clean = metrics.get('acc_clean', [])
    acc_forget = metrics.get('acc_forget', [])

    phase_colors = {'warmup': 'blue', 'standard': 'green', 'detection': 'red'}
    phase_markers = {'warmup': 'o', 'standard': 's', 'detection': '^'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    metrics_data = [
        ('Acc (All)', acc_all),
        ('Acc (Clean)', acc_clean),
        ('Acc (Forget)', acc_forget),
    ]

    for idx, (title_metric, data) in enumerate(metrics_data):
        ax = axes[idx]
        for phase_name, color in phase_colors.items():
            mask = [i for i, p in enumerate(phases) if p == phase_name]
            if mask:
                ax.scatter([m+1 for m in mask], [data[m] for m in mask],
                          c=color, marker=phase_markers[phase_name],
                          label=phase_name, s=50, alpha=0.7)
        ax.plot(range(1, rounds+1), data, 'k-', alpha=0.3, linewidth=1)
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title_metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  阶段对比图已保存: {save_path}")


def plot_final_summary(metrics, save_path, title="Experiment Summary"):
    """生成最终实验总结仪表盘"""
    if not metrics.get('round'):
        return

    setup_plot_style()
    rounds = len(metrics['round'])
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    rounds_arr = np.array(metrics['round'])

    # 1. 主准确率曲线
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rounds_arr, metrics['acc_all'], 'b-o', label='Acc (All)', linewidth=2, markersize=4)
    ax1.plot(rounds_arr, metrics['acc_clean'], 'g-s', label='Acc (Clean)', linewidth=2, markersize=4)
    ax1.plot(rounds_arr, metrics['acc_forget'], 'r-^', label='Acc (Forget)', linewidth=2, markersize=4)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. p_c 演化
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(metrics.get('p_c_mal_from', []), 'r-o', label='p_c (mal_from)', markersize=4)
    ax2.plot(metrics.get('p_c_legit_from', []), 'g-s', label='p_c (legit_from)', markersize=4)
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Detection Round')
    ax2.set_ylabel('p_c')
    ax2.set_title('Pollution Score (p_c)')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 疑似客户端数量
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(1, len(metrics.get('n_suspects', []))+1), 
            metrics.get('n_suspects', []), color='orange', alpha=0.7)
    ax3.set_xlabel('Detection Round')
    ax3.set_ylabel('# Suspects')
    ax3.set_title('Detected Suspects per Round')
    ax3.grid(True, alpha=0.3)

    # 4. 检测混淆
    ax4 = fig.add_subplot(gs[1, 2])
    tp = [len(d) for d in metrics.get('detected_mal', [])]
    fp = [len(f) for f in metrics.get('false_pos', [])]
    fn = [len(f) for f in metrics.get('false_neg', [])]
    x = np.arange(len(tp))
    width = 0.25
    ax4.bar(x - width, tp, width, label='TP', color='green', alpha=0.7)
    ax4.bar(x, fp, width, label='FP', color='orange', alpha=0.7)
    ax4.bar(x + width, fn, width, label='FN', color='red', alpha=0.7)
    ax4.set_xlabel('Detection Round')
    ax4.set_ylabel('Count')
    ax4.set_title('Detection Confusion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 最终逐类准确率
    ax5 = fig.add_subplot(gs[2, :])
    per_class = metrics.get('per_class', [])
    if per_class:
        final_pc = per_class[-1]
        classes = list(range(10))
        accs = [final_pc.get(c, 0) for c in classes]
        colors = ['red' if c in (5, 7) else 'green' if c in (3, 5) else 'blue' for c in classes]
        bars = ax5.bar(classes, accs, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Final Per-Class Accuracy')
        ax5.set_ylim(0, 105)
        for bar, acc in zip(bars, accs):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}', ha='center', va='bottom', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  实验总结仪表盘已保存: {save_path}")


def generate_all_plots(metrics, output_dir):
    """一键生成所有实验结果图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"生成实验结果图表 -> {output_dir}")
    print(f"{'='*50}")

    # 1. 训练曲线
    plot_training_curves(metrics, 
                         os.path.join(output_dir, 'training_curves.png'),
                         title="LabelModifyFL Training Curves")

    # 2. 逐类热力图
    if metrics.get('per_class'):
        plot_per_class_heatmap(metrics['per_class'],
                               os.path.join(output_dir, 'per_class_heatmap.png'),
                               title="Per-Class Accuracy Heatmap")

    # 3. 检测混淆矩阵
    if metrics.get('detected_mal'):
        plot_detection_confusion(
            metrics.get('detected_mal', []),
            metrics.get('false_pos', []),
            metrics.get('false_neg', []),
            os.path.join(output_dir, 'detection_confusion.png'),
            title="Detection Confusion per Round"
        )

    # 4. p_c 演化
    if metrics.get('p_c_mal_from'):
        plot_p_c_evolution(
            metrics['p_c_mal_from'],
            metrics['p_c_legit_from'],
            os.path.join(output_dir, 'p_c_evolution.png'),
            title="Pollution Score (p_c) Evolution"
        )

    # 5. 阶段对比
    if metrics.get('phase'):
        plot_detection_phase_comparison(
            metrics,
            os.path.join(output_dir, 'phase_comparison.png'),
            title="Phase-wise Performance Comparison"
        )

    # 6. 总结仪表盘
    plot_final_summary(metrics,
                       os.path.join(output_dir, 'experiment_summary.png'),
                       title="LabelModifyFL Experiment Summary")

    print(f"\n{'='*50}")
    print("所有图表生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # 简单测试
    test_metrics = {
        'round': list(range(1, 21)),
        'acc_all': [10, 20, 35, 45, 52, 58, 62, 65, 67, 68, 69, 70, 71, 71, 72, 72, 72, 73, 73, 73],
        'acc_clean': [10, 20, 35, 45, 53, 59, 63, 66, 68, 69, 70, 71, 72, 72, 73, 73, 73, 74, 74, 74],
        'acc_forget': [90, 80, 60, 45, 35, 30, 28, 25, 22, 20, 18, 15, 14, 12, 11, 10, 9, 8, 8, 7],
        'n_suspects': [0,0,0,0,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        'p_c_mal_from': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.1],
        'p_c_legit_from': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'detected_mal': [[0], [0], [0], [0], [0,1], [], [], [], [], [0]],
        'false_pos': [[], [], [], [], [], [], [], [], [], []],
        'false_neg': [[], [], [], [], [], [], [], [], [], []],
        'per_class': [{i: 10+5*i for i in range(10)} for _ in range(20)],
        'phase': ['warmup']*4 + ['detection']*4 + ['standard']*4 + ['detection']*4 + ['standard']*4,
        'round': list(range(1, 21))
    }
    
    generate_all_plots(test_metrics, './test_plots')
    print("测试完成！")