使用示例
========

基本操作
--------

1. 加载数据集
~~~~~~~~~~~~

.. code-block:: python

    from cocox import COCOX, CCX
    from pathlib import Path

    # 从JSON文件加载
    cocox = COCOX("path/to/annotations/instances_train.json")

    # 使用自定义配置
    cfg = CCX(
        ROOT=Path("dataset"),
        ANNDIR="annotations",
        IMGDIR="images",
        ANNFILE="instances_train.json"
    )
    cocox = COCOX(cfg=cfg)

    # 从字典创建
    data_dict = {
        "images": [
            {"id": 1, "file_name": "test.jpg", "height": 100, "width": 100}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30]}
        ],
        "categories": [
            {"id": 1, "name": "test", "supercategory": ""}
        ]
    }
    cocox = COCOX(data=data_dict)

2. 数据集合并
~~~~~~~~~~~~

.. code-block:: python

    # 准备两个数据集
    train_set = COCOX("dataset1/annotations/instances_train.json")
    val_set = COCOX("dataset2/annotations/instances_val.json")

    # 合并数据集
    merged = train_set.merge(
        others=[val_set],
        cat_keep=True,  # 保持原始类别ID
        dst_file=CCX(
            ROOT="merged_dataset",
            ANNFILE="instances_merged.json"
        ),
        save_img=True  # 同时复制图片
    )

    # 合并多个数据集
    datasets = [
        "dataset1/annotations/instances_train.json",
        "dataset2/annotations/instances_train.json",
        "dataset3/annotations/instances_train.json"
    ]

    # 逐个合并
    base_set = COCOX(datasets[0])
    for dataset in datasets[1:]:
        other_set = COCOX(dataset)
        base_set = base_set.merge(
            others=[other_set],
            cat_keep=True
        )

3. 数据集分割
~~~~~~~~~~~~

.. code-block:: python

    # 按比例分割数据集
    splits = cocox.split(
        ratio=[0.7, 0.2, 0.1],  # 训练集:验证集:测试集
        ratio_name=["train", "val", "test"],
        dst_file=CCX(ROOT="split_dataset")
    )

    # 获取分割后的数据集
    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"]

    # 按文件分割
    splits = cocox.split(
        ratio=[0.7, 0.2, 0.1],
        by_file=True,
        ratio_name=["train", "val", "test"]
    )

4. 数据过滤
~~~~~~~~~~

.. code-block:: python

    # 按类别过滤
    person_car = cocox.filter(
        cats=["person", "car"],
        mod="or"  # 包含person或car的数据
    )

    # 按图片过滤
    specific_imgs = cocox.filter(
        imgs=["image1.jpg", "image2.jpg"]
    )

    # 复杂过滤
    filtered = cocox.filter(
        cats=["person"],
        imgs=["image1.jpg"],
        mod="and",  # 同时满足两个条件
        level="ann",  # 在标注级别过滤
        keep_empty_img=False  # 不保留没有标注的图片
    )

    # 反向过滤
    not_person = cocox.filter(
        cats=["person"],
        revert=True  # 获取不包含person的数据
    )

5. 类别管理
~~~~~~~~~~

.. code-block:: python

    # 更新类别ID和名称
    cocox.update_cat({
        1: "person",
        2: "car",
        3: "bike"
    })

    # 重命名类别
    cocox.rename_cat("bicycle", "bike")

    # 对齐两个数据集的类别
    other_categories = {
        1: "person",
        2: "vehicle",
        3: "animal"
    }
    mapping = cocox.align_cat(other_categories)

    # 强制更新类别
    cocox.update_cat_force({
        1: "person",
        2: "car"
    })

6. 可视化
~~~~~~~~

.. code-block:: python

    # 可视化数据集统计信息
    cocox.vis_anno_info(save_dir="vis_output")

    # 可视化标注结果
    cocox.vis_gt(
        dst_dir="vis_output/annotations",
        overwrite=True
    )

    # 可视化特定图片的标注
    cocox.vis_gt(
        src_path=["image1.jpg", "image2.jpg"],
        dst_dir="vis_output/specific",
        overwrite=True
    )

7. 格式转换
~~~~~~~~~~

.. code-block:: python

    # 转换为YOLO格式
    cocox.save_yolo("yolo_dataset")

    # 保存完整数据集
    cocox.save_data(
        dst_file=CCX(
            ROOT="output_dataset",
            ANNFILE="instances_processed.json"
        ),
        visual=True,  # 同时保存可视化结果
        yolo=True,    # 同时保存YOLO格式
        overwrite=True
    )

    # 只保存标注文件
    cocox.save_data(
        dst_file=CCX(
            ROOT="output_dataset",
            ANNFILE="instances_processed.json"
        ),
        only_ann=True
    )

高级用例
--------

1. 数据校正
~~~~~~~~~~

.. code-block:: python

    # 在加载时校正数据
    cocox = COCOX(
        "path/to/dataset.json",
        correct_data=True
    )

    # 自定义校正函数
    def correct_size(img, ann):
        # 修正图片尺寸
        img["width"] = 1920
        img["height"] = 1080
        return img, ann

    # 应用校正
    cocox.correct(
        callback=correct_size,
        dst_file=CCX(ROOT="corrected_dataset")
    )

    # 复杂校正示例
    def complex_correction(img, ann):
        # 修正图片信息
        if img["width"] > 1920:
            img["width"] = 1920
        if img["height"] > 1080:
            img["height"] = 1080
            
        # 修正标注信息
        if ann["bbox"][0] < 0:
            ann["bbox"][0] = 0
        if ann["bbox"][1] < 0:
            ann["bbox"][1] = 0
            
        return img, ann

2. 批量处理
~~~~~~~~~~

.. code-block:: python

    # 处理多个数据集
    datasets = [
        "dataset1/annotations/instances_train.json",
        "dataset2/annotations/instances_train.json",
        "dataset3/annotations/instances_train.json"
    ]

    # 合并所有数据集
    base_set = COCOX(datasets[0])
    for dataset in datasets[1:]:
        other_set = COCOX(dataset)
        base_set = base_set.merge(
            others=[other_set],
            cat_keep=True
        )

    # 过滤并分割
    filtered = base_set.filter(cats=["person", "car"])
    splits = filtered.split(ratio=[0.8, 0.2])

    # 保存处理结果
    for name, split_data in splits.items():
        split_data.save_data(
            dst_file=CCX(
                ROOT=f"output/{name}",
                ANNFILE=f"instances_{name}.json"
            ),
            visual=True,
            yolo=True
        )

3. 数据集统计
~~~~~~~~~~~~

.. code-block:: python

    # 获取数据集统计信息
    stats = cocox.static(save=True, static_path="stats.json")

    # 打印统计信息
    print(f"总图片数: {stats['imgs']}")
    print(f"总标注数: {stats['anns']}")
    print(f"类别统计: {stats['cats']}")
    print(f"标注图片数: {stats['img_in_ann']}")
    print(f"文件夹图片数: {stats.get('img_in_folder', 0)}")

4. 完整工作流程示例
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cocox import COCOX, CCX
    from pathlib import Path

    # 1. 加载数据集
    cocox = COCOX("path/to/dataset/annotations/instances_train.json")

    # 2. 数据过滤
    filtered = cocox.filter(
        cats=["person", "car"],
        mod="or"
    )

    # 3. 更新类别
    filtered.update_cat({
        1: "person",
        2: "car"
    })

    # 4. 分割数据集
    splits = filtered.split(
        ratio=[0.7, 0.2, 0.1],
        ratio_name=["train", "val", "test"]
    )

    # 5. 处理每个分割
    for name, split_data in splits.items():
        # 5.1 校正数据
        def correct_data(img, ann):
            # 确保标注框在图片范围内
            if ann["bbox"][0] < 0:
                ann["bbox"][0] = 0
            if ann["bbox"][1] < 0:
                ann["bbox"][1] = 0
            return img, ann

        corrected = split_data.correct(
            callback=correct_data,
            dst_file=CCX(ROOT=f"output/{name}")
        )

        # 5.2 保存结果
        corrected.save_data(
            dst_file=CCX(
                ROOT=f"output/{name}",
                ANNFILE=f"instances_{name}.json"
            ),
            visual=True,  # 保存可视化结果
            yolo=True,    # 保存YOLO格式
            overwrite=True
        )

        # 5.3 生成统计信息
        stats = corrected.static(
            save=True,
            static_path=f"output/{name}/stats.json"
        ) 