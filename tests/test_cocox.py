#!/usr/bin/env python

"""Tests for `cocox` package."""

import pytest
import tempfile, json
from cocox import COCOX,CCX
from pathlib import Path

@pytest.fixture
def sample_cc():
    """创建一个用于测试的 CocoX 实例"""
    # 这里使用一个小型的 COCO 格式数据集路径
    # 根据你的实际情况修改路径
    coco_json_path = "tests/data/sample_coco.json"
    return COCOX(coco_json_path)

def test_init_with_dict():
    """测试使用字典初始化COCOX"""
    sample_coco_dict = {
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
    ccocox = COCOX(data=sample_coco_dict)
    assert ccocox.data is not None
    assert len(ccocox.data.dataset['images']) == 1
    assert len(ccocox.data.dataset['annotations']) == 1
    assert len(ccocox.data.dataset['categories']) == 1

def test_init_with_coco(sample_cc):
    """测试使用COCO对象初始化COCOX"""
    ccocox = COCOX(data=sample_cc)
    assert ccocox.data is not None
    assert len(ccocox.data.dataset['images']) == 1
    assert len(ccocox.data.dataset['annotations']) == 5
    assert len(ccocox.data.dataset['categories']) == 3

def test_init_with_cocox():
    """测试使用COCOX对象初始化COCOX"""
    sample_coco_dict = {
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
    ccocox1 = COCOX(data=sample_coco_dict)
    ccocox2 = COCOX(data=ccocox1)
    assert ccocox2.data is not None
    assert len(ccocox2.data.dataset['images']) == 1
    assert ccocox1.cfg == ccocox2.cfg

def test_init_with_path():
    """测试使用路径初始化COCOX"""   
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        # 创建必要的目录结构
        ann_dir = temp_dir / "annotations"
        ann_dir.mkdir()
        
        # 创建一个简单的COCO格式文件
        coco_file = ann_dir / "instances_default.json"
        with open(coco_file, "w") as f:
            json.dump({
                "images": [],
                "annotations": [],
                "categories": []
            }, f)
            
        ccocox = COCOX(data=coco_file)
        assert ccocox.data is not None
        assert ccocox.cfg.ROOT == temp_dir
        assert ccocox.cfg.ANNDIR == Path("annotations")
        assert ccocox.cfg.ANNFILE == Path("instances_default.json")

def test_init_with_cfg():
    """测试使用自定义配置初始化COCOX"""    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        custom_cfg = CCX(
            ROOT=temp_dir,
            ANNDIR=Path("custom_annotations"),
            IMGDIR=Path("custom_images"),
            ANNFILE=Path("custom.json")
        )
        ccocox = COCOX(cfg=custom_cfg)
        assert ccocox.cfg.ROOT == temp_dir
        assert ccocox.cfg.ANNDIR == Path("custom_annotations") 
        assert ccocox.cfg.IMGDIR == Path("custom_images")
        assert ccocox.cfg.ANNFILE == Path("custom.json")

def test_init_with_nonexistent_path():
    """测试使用不存在的路径初始化COCOX"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "newdir"
        ccocox = COCOX(data=temp_dir)
        assert ccocox.data is None
        assert ccocox.cfg.ROOT == temp_dir
        assert temp_dir.exists()

def test_init_with_data_and_cfg():
    """测试同时使用data和cfg初始化COCOX"""
    from pathlib import Path
       
    sample_coco_dict = {
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
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        custom_cfg = CCX(
            ROOT=temp_dir,
            ANNDIR=Path("custom_annotations")
        )
        ccocox = COCOX(data=sample_coco_dict, cfg=custom_cfg)
        assert ccocox.data is not None
        assert ccocox.cfg.ROOT == temp_dir
        assert ccocox.cfg.ANNDIR == Path("custom_annotations")
        assert len(ccocox.data.dataset['images']) == 1 

def test_init_test01():
    """测试使用图片目录初始化COCOX"""
    #images 没有子目录
    coco_path0 = "tests/data/test01/annotations/instances_default.json"
    co2x0 = COCOX(coco_path0)
    assert co2x0.cfg.IMGFOLDER == Path(".")
    assert co2x0.cfg.IMGDIR == Path("images")
    assert co2x0.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test01/images").absolute()
    
    
    # images 有子目录
    coco_path1 = "tests/data/test02/annotations/instances_default.json"
    co2x1 = COCOX(coco_path1)
    assert co2x1.cfg.IMGFOLDER == Path("default")
    assert co2x1.cfg.IMGDIR == Path("images")
    assert co2x1.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test02/images/default").absolute()
    
    # 复制没有子目录
    co2x2 = COCOX(co2x0)
    assert co2x2.cfg.IMGFOLDER == Path(".")
    assert co2x2.cfg.IMGDIR == Path("images")
    assert co2x2.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test01/images").absolute()
    
    # 复制有子目录
    co2x3 = COCOX(co2x1)
    assert co2x3.cfg.IMGFOLDER == Path("default")
    assert co2x3.cfg.IMGDIR == Path("images")
    assert co2x3.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test02/images/default").absolute()
    
    # 复制没有子目录，但是有cfg
    c2x4 = CCX(
        IMGDIR=Path("images"),
        IMGFOLDER=Path("default")
    )
    co2x4 = COCOX(co2x0,cfg=c2x4)
    # 原始配置不能被修改
    assert co2x0.cfg.IMGFOLDER == Path(".")
    assert co2x0.cfg.IMGDIR == Path("images")
    assert co2x0.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test01/images").absolute()
    # 新配置生效
    assert co2x4.cfg.IMGFOLDER == Path("default")
    assert co2x4.cfg.IMGDIR == Path("images")
    assert co2x4.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test01/images").absolute()
    
    
    # 复制子目录，但是有cfg
    c2x5 = CCX(
        IMGDIR=Path("images"),
        IMGFOLDER=Path(".")
    )
    co2x5 = COCOX(co2x1,cfg=c2x5)
    # 原始配置不能被修改
    assert co2x1.cfg.IMGFOLDER == Path("default")
    assert co2x1.cfg.IMGDIR == Path("images")
    assert co2x1.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test02/images/default").absolute()
    # 新配置生效
    assert co2x5.cfg.IMGFOLDER == Path(".")
    assert co2x5.cfg.IMGDIR == Path("images")
    assert co2x5.cfg.IMGDIR_SRC.absolute() == Path("tests/data/test02/images/default").absolute()


def test_vis_anno_info():
    """测试标注基础信息接口"""
    coco_path0 = "tests/data/test01/annotations/instances_default.json"
    co2x0 = COCOX(coco_path0)
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        static_data = co2x0.vis_anno_info(save_dir=Path(temp_dir))
        assert static_data is not None
        with open("tests/data/test01/gt_summary.json") as f:
            gt_summary = json.load(f)
            
        assert static_data == gt_summary
    
    
    