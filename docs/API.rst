API 文档
========

COCOX 类
--------

COCOX类是该库的核心类，提供了处理COCO格式数据集的各种功能。它支持多种数据源导入、数据集操作、类别管理、可视化和格式转换等功能。

初始化
~~~~~~~

.. code-block:: python

    COCOX(data=None, cfg=None, **kwargs)

参数:
    - data: 可选，支持以下类型：
        - dict: COCO格式的字典数据
        - COCO: pycocotools.COCO对象
        - COCOX: COCOX对象
        - Path/str: JSON文件路径
    - cfg: 可选，CCX对象，用于配置数据集路径等
    - kwargs:
        - correct_data: bool，是否校正数据，默认为False
        - save_static: bool，是否保存统计信息，默认为False
        - static_path: Path，统计信息保存路径

示例:
    .. code-block:: python

        # 从JSON文件加载
        cocox = COCOX("path/to/annotations/instances_train.json")

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

数据集操作
~~~~~~~~~~

merge
^^^^^

合并多个数据集。支持合并不同来源的数据集，并可以控制类别ID的保持和文件覆盖行为。

.. code-block:: python

    merge(others, cat_keep=None, overwrite=None, dst_file=None, save_img=True)

参数:
    - others: COCOX对象或COCOX对象列表，要合并的数据集
    - cat_keep: bool，是否保留原始类别ID，默认为None
    - overwrite: bool，是否覆盖已存在的文件，默认为None
    - dst_file: CCX，目标文件配置，默认为None
    - save_img: bool，是否保存图片，默认为True

返回:
    - COCOX对象，合并后的数据集

示例:
    .. code-block:: python

        # 合并两个数据集
        merged = cocox1.merge([cocox2, cocox3])

        # 合并时指定参数
        merged = cocox1.merge(
            others=[cocox2, cocox3],
            cat_keep=True,      # 保持原有类别ID
            overwrite=True,     # 覆盖已存在的文件
            dst_file=CCX(...),  # 输出配置
            save_img=True       # 保存图片
        )

split
^^^^^

分割数据集。支持按比例分割和按文件分割两种模式。

.. code-block:: python

    split(ratio=[0.7, 0.2, 0.1], by_file=False, dst_file=None, ratio_name=None, merge=False)

参数:
    - ratio: List[float]，分割比例，默认为[0.7, 0.2, 0.1]
    - by_file: bool，是否按文件分割，默认为False
    - dst_file: CCX，目标文件配置，默认为None
    - ratio_name: List[str]，分割后的名称列表，默认为None
    - merge: bool，是否合并结果，默认为False

返回:
    - Dict[str, COCOX]，分割后的数据集字典

示例:
    .. code-block:: python

        # 按比例分割
        splits = cocox.split(
            ratio=[0.7, 0.2, 0.1],
            ratio_name=["train", "val", "test"]
        )

        # 按文件分割
        splits = cocox.split(
            ratio=[0.7, 0.2, 0.1],
            by_file=True
        )

filter
^^^^^^

过滤数据集。支持按类别、图片、标注ID进行过滤，并支持多种过滤模式。

.. code-block:: python

    filter(cats=[], imgs=[], annIds=[], mod="and", level="img", revert=False, 
           dst_file=None, alignCat=True, keep_all_img=False, keep_empty_img=True)

参数:
    - cats: List[Union[int,str]]，类别列表，默认为[]
    - imgs: List[Union[int,str]]，图片列表，默认为[]
    - annIds: List[int]，标注ID列表，默认为[]
    - mod: str，"and"或"or"，过滤模式，默认为"and"
    - level: str，"img"或"ann"，过滤级别，默认为"img"
    - revert: bool，是否反向过滤，默认为False
    - dst_file: CCX，目标文件配置，默认为None
    - alignCat: bool，是否对齐类别，默认为True
    - keep_all_img: bool，是否保留所有图片，默认为False
    - keep_empty_img: bool，是否保留空图片，默认为True

返回:
    - COCOX对象，过滤后的数据集

示例:
    .. code-block:: python

        # 按类别过滤
        filtered = cocox.filter(
            cats=["person", "car"],
            mod="or"
        )

        # 复杂过滤
        filtered = cocox.filter(
            cats=["person"],
            imgs=["image1.jpg"],
            mod="and",
            level="ann",
            keep_empty_img=False
        )

类别管理
~~~~~~~~

update_cat
^^^^^^^^^^

更新类别信息。支持更新类别ID和名称的映射关系。

.. code-block:: python

    update_cat(new_cat)

参数:
    - new_cat: Dict[int,str]，新的类别映射

示例:
    .. code-block:: python

        # 更新类别
        cocox.update_cat({
            1: "person",
            2: "car",
            3: "bike"
        })

rename_cat
^^^^^^^^^^

重命名类别。支持修改类别的名称。

.. code-block:: python

    rename_cat(raw_cat, new_cat)

参数:
    - raw_cat: str，原类别名
    - new_cat: str，新类别名

示例:
    .. code-block:: python

        # 重命名类别
        cocox.rename_cat("bicycle", "bike")

align_cat
^^^^^^^^^

对齐类别。支持将当前数据集的类别与目标类别映射对齐。

.. code-block:: python

    align_cat(other_cat, cat_keep=True)

参数:
    - other_cat: Dict，目标类别映射
    - cat_keep: bool，是否保留原始类别ID，默认为True

返回:
    - Dict，对齐后的类别映射

示例:
    .. code-block:: python

        # 对齐类别
        other_categories = {
            1: "person",
            2: "vehicle",
            3: "animal"
        }
        mapping = cocox.align_cat(other_categories)

可视化和导出
~~~~~~~~~~~~

vis_anno_info
^^^^^^^^^^^^

可视化标注信息。生成数据集的统计信息和可视化图表。

.. code-block:: python

    vis_anno_info(save_dir=Path(""))

参数:
    - save_dir: Path，保存目录，默认为当前目录

示例:
    .. code-block:: python

        # 可视化标注信息
        cocox.vis_anno_info(save_dir="vis_output")

vis_gt
^^^^^^

可视化标注结果。在图片上绘制标注框和类别信息。

.. code-block:: python

    vis_gt(src_path=None, dst_dir=None, overwrite=True)

参数:
    - src_path: Union[Path,str,List[Union[Path,str]]]，源图片路径，默认为None
    - dst_dir: Union[Path,str]，目标目录，默认为None
    - overwrite: bool，是否覆盖，默认为True

示例:
    .. code-block:: python

        # 可视化标注结果
        cocox.vis_gt(
            dst_dir="vis_output/annotations",
            overwrite=True
        )

save_yolo
^^^^^^^^^

保存为YOLO格式。将COCO格式转换为YOLO格式。

.. code-block:: python

    save_yolo(dst_dir=None, overwrite=True)

参数:
    - dst_dir: Union[Path,str]，目标目录，默认为None
    - overwrite: bool，是否覆盖，默认为True

示例:
    .. code-block:: python

        # 转换为YOLO格式
        cocox.save_yolo("yolo_dataset")

save_data
^^^^^^^^^

保存数据集。支持保存完整数据集、可视化结果和YOLO格式。

.. code-block:: python

    save_data(dst_file=None, visual=False, yolo=False, only_ann=False, overwrite=True)

参数:
    - dst_file: CCX，目标文件配置，默认为None
    - visual: bool，是否保存可视化结果，默认为False
    - yolo: bool，是否保存YOLO格式，默认为False
    - only_ann: bool，是否只保存标注，默认为False
    - overwrite: bool，是否覆盖，默认为True

返回:
    - COCOX对象，保存后的数据集

示例:
    .. code-block:: python

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

CCX 类
------

CCX类用于管理COCO数据集的配置信息。它提供了灵活的数据集路径和文件配置选项。

.. code-block:: python

    CCX(ROOT=Path("."), ANNDIR=Path("annotations"), IMGDIR=Path("images"), 
        ANNFILE=Path("instances_default.json"), IMGFOLDER=Path("."))

参数:
    - ROOT: Path，数据集根目录，默认为当前目录
    - ANNDIR: Path，标注文件目录名，默认为"annotations"
    - IMGDIR: Path，图片目录名，默认为"images"
    - ANNFILE: Path，标注文件名，默认为"instances_default.json"
    - IMGFOLDER: Path，图片子目录名，默认为当前目录

示例:
    .. code-block:: python

        # 基本配置
        cfg = CCX(
            ROOT="path/to/dataset",
            ANNDIR="annotations",
            IMGDIR="images",
            ANNFILE="instances_train.json"
        )

        # 带子目录的配置
        cfg = CCX(
            ROOT="path/to/dataset",
            ANNDIR="annotations",
            IMGDIR="images",
            ANNFILE="instances_train.json",
            IMGFOLDER="train"
        ) 