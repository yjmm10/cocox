API 文档
========

COCOX 类
--------

COCOX类是该库的核心类，提供了处理COCO格式数据集的各种功能。

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
        - correct_data: bool，是否校正数据
        - save_static: bool，是否保存统计信息
        - static_path: Path，统计信息保存路径

数据集操作
~~~~~~~~~~

merge
^^^^^

合并多个数据集。

.. code-block:: python

    merge(others, cat_keep=None, overwrite=None, dst_file=None, save_img=True)

参数:
    - others: COCOX对象或COCOX对象列表
    - cat_keep: bool，是否保留原始类别ID
    - overwrite: bool，是否覆盖已存在的文件
    - dst_file: CCX，目标文件配置
    - save_img: bool，是否保存图片

split
^^^^^

分割数据集。

.. code-block:: python

    split(ratio=[0.7, 0.2, 0.1], by_file=False, dst_file=None, ratio_name=None, merge=False)

参数:
    - ratio: 分割比例
    - by_file: bool，是否按文件分割
    - dst_file: CCX，目标文件配置
    - ratio_name: 分割后的名称列表
    - merge: bool，是否合并结果

filter
^^^^^^

过滤数据集。

.. code-block:: python

    filter(cats=[], imgs=[], annIds=[], mod="and", level="img", revert=False, dst_file=None, alignCat=True, keep_all_img=False, keep_empty_img=True)

参数:
    - cats: 类别列表
    - imgs: 图片列表
    - annIds: 标注ID列表
    - mod: "and"或"or"，过滤模式
    - level: "img"或"ann"，过滤级别
    - revert: bool，是否反向过滤
    - dst_file: CCX，目标文件配置
    - alignCat: bool，是否对齐类别
    - keep_all_img: bool，是否保留所有图片
    - keep_empty_img: bool，是否保留空图片

类别管理
~~~~~~~~

update_cat
^^^^^^^^^^

更新类别信息。

.. code-block:: python

    update_cat(new_cat)

参数:
    - new_cat: Dict[int,str]，新的类别映射

rename_cat
^^^^^^^^^^

重命名类别。

.. code-block:: python

    rename_cat(raw_cat, new_cat)

参数:
    - raw_cat: str，原类别名
    - new_cat: str，新类别名

align_cat
^^^^^^^^^

对齐类别。

.. code-block:: python

    align_cat(other_cat, cat_keep=True)

参数:
    - other_cat: Dict，目标类别映射
    - cat_keep: bool，是否保留原始类别ID

可视化和导出
~~~~~~~~~~~~

vis_anno_info
^^^^^^^^^^^^

可视化标注信息。

.. code-block:: python

    vis_anno_info(save_dir=Path(""))

参数:
    - save_dir: 保存目录

vis_gt
^^^^^^

可视化标注结果。

.. code-block:: python

    vis_gt(src_path=None, dst_dir=None, overwrite=True)

参数:
    - src_path: 源图片路径
    - dst_dir: 目标目录
    - overwrite: bool，是否覆盖

save_yolo
^^^^^^^^^

保存为YOLO格式。

.. code-block:: python

    save_yolo(dst_dir=None, overwrite=True)

参数:
    - dst_dir: 目标目录
    - overwrite: bool，是否覆盖

save_data
^^^^^^^^^

保存数据集。

.. code-block:: python

    save_data(dst_file=None, visual=False, yolo=False, only_ann=False, overwrite=True)

参数:
    - dst_file: CCX，目标文件配置
    - visual: bool，是否保存可视化结果
    - yolo: bool，是否保存YOLO格式
    - only_ann: bool，是否只保存标注
    - overwrite: bool，是否覆盖

CCX 类
------

CCX类用于管理COCO数据集的配置信息。

.. code-block:: python

    CCX(ROOT=Path("."), ANNDIR=Path("annotations"), IMGDIR=Path("images"), 
        ANNFILE=Path("instances_default.json"), IMGFOLDER=Path("."))

参数:
    - ROOT: 数据集根目录
    - ANNDIR: 标注文件目录名
    - IMGDIR: 图片目录名
    - ANNFILE: 标注文件名
    - IMGFOLDER: 图片子目录名 