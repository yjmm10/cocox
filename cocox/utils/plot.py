from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import Colors,STATIC_DATA

colors = Colors()

def xywh2xyxy(x):
    """将边界框坐标从(x,y,width,height)格式转换为(x1,y1,x2,y2)格式"""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def plot_anno_info(boxes, cls, names=(), save_dir=Path("")):
    """
    绘制数据集标签的统计信息图表，用于可视化分析数据集的标签分布情况
    
    功能说明:
        1. 生成类别分布直方图，显示每个类别的实例数量
        2. 创建边界框可视化图，展示边界框的形状和大小分布
        3. 生成边界框坐标散点图，分析x,y坐标的分布情况
        4. 生成边界框宽高散点图，分析宽度和高度的分布关系
        5. 保存所有图表到指定目录
    
    参数:
        boxes (numpy.ndarray): 边界框数据，格式为 [x, y, width, height],归一化到0-1
        cls (numpy.ndarray): 类别标签数组，从0开始
        names (dict, optional): 类别ID到类别名称的映射字典，默认为空元组
        save_dir (Path, optional): 保存图表的目录路径，默认为当前目录
    
    返回:
        None: 函数将图表保存到指定目录，不返回值
    """
    """绘制数据集标签的统计信息"""
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制数据集标签
    nc = int(cls.max() + 1)  # 类别数量
    boxes = boxes[:1000000]  # 限制为100万个框
    x = pd.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn相关图
    sns.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "boxes_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib标签图
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color(colors(i,format="float"))
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    sns.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # 绘制边界框分布可视化图
    boxes_vis = boxes.copy()
    boxes_vis[:, 0:2] = 0.5  # center
    boxes_vis = xywh2xyxy(boxes_vis) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for tmp_cls, box in zip(cls[:500], boxes_vis[:500]):
        ImageDraw.Draw(img).rectangle(box.astype(np.int32).tolist(), width=1, outline=colors(tmp_cls,format="int"))
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / "boxes_summary.jpg", dpi=200)
    plt.close()
    
    # 添加类别边界框分布图
    categories = sorted(set(cls.tolist()))
    n_cats = len(categories)
    
    if n_cats > 0:
        rows = int(np.ceil(n_cats / 3))
        fig, axs = plt.subplots(rows, 3, figsize=(15, rows * 4))
        axs = axs.flatten() if rows > 1 else [axs] if n_cats == 1 else axs
        
        for i, cat_id in enumerate(categories):
            if i < len(axs):
                cat_boxes = boxes_vis[cls == cat_id]
                if len(cat_boxes):
                    cat_img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
                    draw = ImageDraw.Draw(cat_img)
                    
                    for box in cat_boxes[:1000]:
                        draw.rectangle(box.astype(np.int32).tolist(), width=1, outline=colors(cat_id,format="int"))
                    
                    axs[i].imshow(cat_img)
                    cat_name = names.get(cat_id+1, f"Class {cat_id}")
                    # 将类别名称显示在x轴下方
                    # 显示左右侧坐标轴，标签在x轴下方
                    axs[i].spines['left'].set_visible(True)
                    axs[i].spines['right'].set_visible(False)
                    axs[i].spines['bottom'].set_visible(True)
                    axs[i].spines['top'].set_visible(False)
                    axs[i].xaxis.set_ticks_position('bottom')
                    axs[i].yaxis.set_ticks_position('left')
                    # 保留x轴刻度，与y轴刻度保持一致
                    axs[i].set_xticks([0, 250, 500, 750, 1000])
                    axs[i].set_yticks([0, 250, 500, 750, 1000])
                    # 确保刻度可见
                    axs[i].tick_params(axis='both', which='both', length=5, width=1, direction='out')
                    axs[i].set_xlabel(f"{cat_name} (n={len(cat_boxes)})")
        
        for i in range(len(categories), len(axs)):
            axs[i].axis("off")
            
        plt.tight_layout()
        plt.savefig(save_dir / "class_distribution.jpg", dpi=200)
        plt.close()
           
def plot_summary(static_data:STATIC_DATA, save_dir=Path(""),show_num_cat=20):
    """
    Visualize dataset statistics, generating a comprehensive chart with multiple sections
    
    Parameters:
        static_data: STATIC_DATA object containing dataset statistics
        save_dir: Directory path to save the chart, default is current directory
    """
    import json
    
    assert isinstance(static_data, STATIC_DATA), f"static_data must be a STATIC_DATA object, but got {type(static_data)}"
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 将所有数据保存到JSON文件 - 完整数据保存
    summary_data = {}
    for attr_name in dir(static_data):
        if not attr_name.startswith('__') and not callable(getattr(static_data, attr_name)):
            attr_value = getattr(static_data, attr_name)
            summary_data[attr_name] = attr_value
    
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 自动分类变量
    nums_vars = {}
    diff_vars = {}
    relate_vars = {}
    
    for attr_name in dir(static_data):
        if attr_name.startswith('nums_') and 'diff' not in attr_name and not callable(getattr(static_data, attr_name)):
            nums_vars[attr_name] = getattr(static_data, attr_name)
        elif attr_name.startswith('nums_') and 'diff' in attr_name and not callable(getattr(static_data, attr_name)):
            diff_vars[attr_name] = getattr(static_data, attr_name)
        elif attr_name.startswith('relate_') and not callable(getattr(static_data, attr_name)):
            relate_vars[attr_name] = getattr(static_data, attr_name)
    
    # 根据变量数量确定图表布局
    num_relate = len(relate_vars)
    total_plots = 2 + (num_relate if num_relate > 0 else 1)  # Basic stats + Diff stats + Relation vars (at least 1)
    
    # 创建紧凑的图表布局
    rows = max(2, (total_plots + 1) // 2)
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(14, 4 * rows), constrained_layout=True)
    
    # 确保axs是二维数组
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = np.array([[ax] for ax in axs])
    
    # 1. 基础统计数据 (nums变量) - 左上角
    ax1 = axs[0, 0]
    if nums_vars:
        # 转换变量名为显示标签
        display_labels = {
            key: ' '.join(key.replace('nums_', '').split('_')).title() 
            for key in nums_vars.keys()
        }
        
        labels = list(display_labels.values())
        values = list(nums_vars.values())
        
        # 使用普通的matplotlib条形图
        bars = ax1.bar(range(len(values)), values, color='skyblue')
        ax1.set_title('Dataset Basic Statistics', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Count')
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 设置y轴从0开始
        ax1.set_ylim(0, max(values) * 1.15)
    else:
        ax1.text(0.5, 0.5, 'No basic statistics available', ha='center', va='center')
        ax1.set_title('Dataset Basic Statistics', fontsize=12, fontweight='bold')
    
    # 2. 差异统计数据 (diff变量) - 右上角
    ax2 = axs[0, 1]
    if diff_vars:
        # 转换变量名为显示标签
        display_labels = {
            key: ' '.join(key.replace('nums_', '').split('_')).title() 
            for key in diff_vars.keys()
        }
        
        labels = list(display_labels.values())
        values = list(diff_vars.values())
        
        # 使用普通的matplotlib条形图
        bars = ax2.bar(range(len(values)), values, color='lightcoral')
        ax2.set_title('Dataset Difference Statistics', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Count')
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 设置y轴从0开始
        max_value = max(values) if values and max(values) > 0 else 1
        ax2.set_ylim(0, max_value * 1.15)
    else:
        ax2.text(0.5, 0.5, 'No difference statistics available', ha='center', va='center')
        ax2.set_title('Dataset Difference Statistics', fontsize=12, fontweight='bold')
    
    # 3. 关系变量 - 动态放置
    plot_index = 0
    for key, value in relate_vars.items():
        # 计算当前图表位置
        row = 1 + plot_index // 2
        col = plot_index % 2
        
        # 检查是否超出了图表范围
        if row < rows and col < cols:
            ax = axs[row, col]
            plot_index += 1
            
            # 转换变量名为显示标签
            display_label = ' '.join(key.replace('relate_', '').split('_')).title()
            
            if value and isinstance(value, dict) and value:
                categories = list(value.keys())
                
                # 处理不同的值类型
                if all(isinstance(v, (int, float)) for v in value.values()):
                    counts = list(value.values())
                elif all(isinstance(v, list) for v in value.values()):
                    counts = [len(v) for v in value.values()]
                else:
                    # 混合类型，尝试智能获取计数
                    counts = []
                    for v in value.values():
                        if isinstance(v, (int, float)):
                            counts.append(v)
                        elif isinstance(v, list):
                            counts.append(len(v))
                        else:
                            counts.append(1)  # 默认计数
                
                # 按计数排序
                sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
                categories, counts = zip(*sorted_data) if sorted_data else ([], [])
                
                # 如果类别太多，只显示前10个
                if len(categories) > show_num_cat:
                    categories = categories[:show_num_cat]
                    counts = counts[:show_num_cat]
                    ax.set_title(f'{display_label} (Top 10)', fontsize=12, fontweight='bold')
                else:
                    ax.set_title(display_label, fontsize=12, fontweight='bold')
                
                # 使用普通的matplotlib条形图，使用colors中的颜色
                color_list = [colors(i,format="float") for i in range(len(categories))]
                color_values = [color_list[i % len(color_list)] for i in range(len(categories))]
                bars = ax.bar(range(len(categories)), counts, color=color_values)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right')
                ax.set_ylabel('Count')
                
                # 在条形上方添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                # 设置y轴从0开始
                ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
            else:
                ax.text(0.5, 0.5, f'No {display_label} data available', ha='center', va='center')
                ax.set_title(display_label, fontsize=12, fontweight='bold')
    
    # 如果没有关系变量，添加一个汇总饼图
    if not relate_vars:
        ax3 = axs[1, 0]
        
        # 合并所有数值数据用于汇总饼图
        all_nums = {**nums_vars, **diff_vars}
        if all_nums:
            # 转换变量名为显示标签
            display_labels = {
                key: ' '.join(key.replace('nums_', '').split('_')).title() 
                for key in all_nums.keys()
            }
            
            pie_data = list(all_nums.values())
            pie_labels = list(display_labels.values())
            
            # 过滤掉零值
            non_zero_indices = [i for i, val in enumerate(pie_data) if val > 0]
            filtered_data = [pie_data[i] for i in non_zero_indices]
            filtered_labels = [pie_labels[i] for i in non_zero_indices]
            
            if filtered_data:
                # 使用饼图显示数据分布
                wedges, texts = ax3.pie(filtered_data, 
                                      startangle=90, 
                                      colors=plt.cm.tab10.colors[:len(filtered_data)])
                
                # 添加图例而不是直接在饼图上标注，使布局更清晰
                ax3.legend(wedges, [f'{l} ({d})' for l, d in zip(filtered_labels, filtered_data)], 
                         loc='center left', bbox_to_anchor=(1, 0.5))
                
                # 添加百分比标签
                total = sum(filtered_data)
                for i, wedge in enumerate(wedges):
                    ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                    x = wedge.r * 0.8 * np.cos(np.deg2rad(ang))
                    y = wedge.r * 0.8 * np.sin(np.deg2rad(ang))
                    percent = filtered_data[i] / total * 100
                    ax3.text(x, y, f'{percent:.1f}%', ha='center', va='center', fontweight='bold')
                
                ax3.set_title('Dataset Summary', fontsize=12, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No summary data available', ha='center', va='center')
                ax3.set_title('Dataset Summary', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No summary data available', ha='center', va='center')
            ax3.set_title('Dataset Summary', fontsize=12, fontweight='bold')
    
    # 隐藏未使用的子图
    for i in range(rows):
        for j in range(cols):
            if i * cols + j >= 2 + plot_index and (i > 0 or j > 0):  # 保留前两个图表
                axs[i, j].axis('off')
    
    # 调整布局，确保图表紧凑但不重叠
    plt.tight_layout()
    plt.savefig(save_dir / "dataset_summary.jpg", dpi=300)
    plt.close()
    return summary_data
