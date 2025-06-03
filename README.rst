=====
COCOX
=====


.. image:: https://img.shields.io/pypi/v/cocox.svg
        :target: https://pypi.python.org/pypi/cocox

.. image:: https://img.shields.io/travis/liferecords/cocox.svg
        :target: https://travis-ci.com/liferecords/cocox

.. image:: https://readthedocs.org/projects/cocox/badge/?version=latest
        :target: https://cocox.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT license
* Documentation: https://cocox.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage




TODO:

单个文件：
可视化 - 单个文件，多个文件，所有文件（默认）
统计 - 数量信息，xywh关联信息


global_info: 存在callback中用于记录信息

img_input:
    imgpath: 图片路径
    cats: 类别信息
..     以下为纠正后的目录信息
    root: 根目录
    imgdir: 图片目录 'images'
    imgfolder: 图片文件夹 'train','val','test'

    file_name: 图片名称
    width: 图片宽度
    height: 图片高度
    
ann_input:
    id: 标注id
    category_id: 类别id
    bbox: 标注框
    segmentation: 分割信息
    area: 面积
    iscrowd: 是否为crowd
    