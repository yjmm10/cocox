# COCOX

COCOX 是一个用于处理和管理COCO格式数据集的Python工具库。它提供了丰富的功能来简化COCO数据集的操作、可视化和转换过程。

## 特性

- 支持多种数据源导入（字典、COCO对象、JSON文件等）
- 数据集合并和分割
- 类别管理（重命名、更新、对齐等）
- 数据过滤和筛选
- 数据集统计和可视化
- YOLO格式转换
- 标注校正和验证

## 安装

```bash
pip install cocox
```

## 基本用法

### 1. 初始化COCOX对象

```python
from cocox import COCOX, CCX

# 从JSON文件加载
cocox = COCOX("path/to/annotations/instances_xxx.json")

# 从字典创建
data_dict = {
    "images": [...],
    "annotations": [...],
    "categories": [...]
}
cocox = COCOX(data=data_dict)

# 使用自定义配置
cfg = CCX(
    ROOT="path/to/dataset",
    ANNDIR="annotations",
    IMGDIR="images",
    ANNFILE="instances_train.json"
)
cocox = COCOX(cfg=cfg)
```

### 2. 数据集操作

```python
# 合并数据集
merged = cocox1.merge([cocox2, cocox3])

# 分割数据集
splits = cocox.split(ratio=[0.7, 0.2, 0.1])

# 过滤数据
filtered = cocox.filter(
    cats=["person", "car"],  # 按类别过滤
    imgs=["img1.jpg", "img2.jpg"],  # 按图片过滤
    mod="and"  # 过滤模式：and/or
)

# 更新类别
cocox.update_cat({1: "person", 2: "car"})

# 重命名类别
cocox.rename_cat("old_name", "new_name")
```

### 3. 可视化和导出

```python
# 可视化标注信息
cocox.vis_anno_info(save_dir="output")

# 可视化标注结果
cocox.vis_gt(dst_dir="output")

# 导出为YOLO格式
cocox.save_yolo("output_dir")

# 保存数据集
cocox.save_data(
    dst_file=CCX(ROOT="output"),
    visual=True,  # 同时保存可视化结果
    yolo=True,    # 同时保存YOLO格式
)
```

## 配置说明

COCOX使用CCX类来管理配置，主要配置项包括：

- ROOT: 数据集根目录
- ANNDIR: 标注文件目录名（相对于ROOT）
- IMGDIR: 图片目录名（相对于ROOT）
- ANNFILE: 标注文件名
- IMGFOLDER: 图片子目录名（如果有）

## 更多文档

- [API文档](docs/API.rst)
- [使用示例](docs/EXAMPLES.rst)

## 许可证

MIT License 