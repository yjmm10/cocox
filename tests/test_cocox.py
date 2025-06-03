#!/usr/bin/env python

"""Tests for `cocox` package."""

import pytest
import tempfile, json
from cocox import COCOX,CCX
from pathlib import Path


TEST_DATA_PATH = Path("tests/data")

def test_ccx():
    # 支持字符串和Path对象
    
    # annfile与imgfolder
    ccx = CCX(ROOT=Path(".")) # ROOT一般必须指定
    assert ccx.IMGFOLDER == Path(".")
    assert ccx.ANNFILE == Path("instances_default.json")
    
    # 指定IMGFOLDER
    ccx = CCX(ROOT=Path("."),IMGFOLDER=Path("default"))
    assert ccx.IMGFOLDER == Path("default")
    assert ccx.ANNFILE == Path("instances_default.json")
    
    # 指定IMGFOLDER为.
    ccx = CCX(ROOT=Path("."),IMGFOLDER=".")
    assert ccx.IMGFOLDER == Path(".")
    assert ccx.ANNFILE == Path("instances_default.json")
    
    # 指定ANNFILE
    ccx = CCX(ROOT=Path("."),ANNFILE="instances_default.json")
    assert ccx.IMGFOLDER == Path("default") 
    assert ccx.ANNFILE == Path("instances_default.json")
    
    # 同时指定IMGFOLDER和ANNFILE
    ccx = CCX(ROOT=Path("."),IMGFOLDER="happy",ANNFILE="instances_default.json")
    assert ccx.IMGFOLDER == Path("happy")
    assert ccx.ANNFILE == Path("instances_default.json")




@pytest.fixture
def sample_cc():
    """创建一个用于测试的 CocoX 实例"""
    # 加载时候一般使用json文件直接加载
    # 当创建一个空对象才使用CCX
    coco_json_path = TEST_DATA_PATH / "sample_coco.json"
    return COCOX(coco_json_path)

@pytest.fixture
def common_cc():
    """创建一个用于测试的 CocoX 实例"""
    # 加载时候一般使用json文件直接加载
    # 当创建一个空对象才使用CCX
    return COCOX(TEST_DATA_PATH / "common/annotations/instances_default.json")


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
    coco_path0 = TEST_DATA_PATH / "test01/annotations/instances_default.json"
    co2x0 = COCOX(coco_path0)
    assert co2x0.cfg.IMGFOLDER == Path(".")
    assert co2x0.cfg.IMGDIR == Path("images")
    assert co2x0.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test01/images").absolute()
    
    
    # images 有子目录
    coco_path1 = TEST_DATA_PATH / "test02/annotations/instances_default.json"
    co2x1 = COCOX(coco_path1)
    assert co2x1.cfg.IMGFOLDER == Path("default")
    assert co2x1.cfg.IMGDIR == Path("images")
    assert co2x1.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test02/images/default").absolute()
    
    # 复制没有子目录
    co2x2 = COCOX(co2x0)
    assert co2x2.cfg.IMGFOLDER == Path(".")
    assert co2x2.cfg.IMGDIR == Path("images")
    assert co2x2.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test01/images").absolute()
    
    # 复制有子目录
    co2x3 = COCOX(co2x1)
    assert co2x3.cfg.IMGFOLDER == Path("default")
    assert co2x3.cfg.IMGDIR == Path("images")
    assert co2x3.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test02/images/default").absolute()
    
    # 复制没有子目录，但是有cfg
    c2x4 = CCX(
        IMGDIR=Path("images"),
        IMGFOLDER=Path("default")
    )
    co2x4 = COCOX(co2x0,cfg=c2x4)
    # 原始配置不能被修改
    assert co2x0.cfg.IMGFOLDER == Path(".")
    assert co2x0.cfg.IMGDIR == Path("images")
    assert co2x0.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test01/images").absolute()
    # 新配置生效
    assert co2x4.cfg.IMGFOLDER == Path("default")
    assert co2x4.cfg.IMGDIR == Path("images")
    assert co2x4.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test01/images").absolute()
    
    
    # 复制子目录，但是有cfg
    c2x5 = CCX(
        IMGDIR=Path("images"),
        IMGFOLDER=Path(".")
    )
    co2x5 = COCOX(co2x1,cfg=c2x5)
    # 原始配置不能被修改
    assert co2x1.cfg.IMGFOLDER == Path("default")
    assert co2x1.cfg.IMGDIR == Path("images")
    assert co2x1.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test02/images/default").absolute()
    # 新配置生效
    assert co2x5.cfg.IMGFOLDER == Path(".")
    assert co2x5.cfg.IMGDIR == Path("images")
    assert co2x5.cfg.IMGDIR_SRC.absolute() == Path(TEST_DATA_PATH / "test02/images/default").absolute()


def test_vis_anno_info():
    """测试标注基础信息接口"""
    coco_path0 = TEST_DATA_PATH / "common/annotations/instances_default.json"
    co2x0 = COCOX(coco_path0)
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = TEST_DATA_PATH / "common1"
        static_data = co2x0.vis_anno_info(save_dir=Path(temp_dir))
        print(static_data)
        assert static_data is not None
        with open(TEST_DATA_PATH / "common/gt_summary.json") as f:
            gt_summary = json.load(f)
            
        # 判断两个字典是否相等
        assert all(static_data[key] == gt_summary[key] for key in static_data.keys())
        assert all(static_data[key] == gt_summary[key] for key in gt_summary.keys())


def test_vis_gt():
    """测试可视化接口"""
    coco_path0 = TEST_DATA_PATH / "test01/annotations/instances_default.json"
    co2x0 = COCOX(coco_path0)
    with tempfile.TemporaryDirectory() as temp_dir:
        co2x0.vis_gt(src_path="a new precession formula(fukushima 2003)_5.png",dst_dir=Path(temp_dir))
        assert Path(temp_dir).joinpath("a new precession formula(fukushima 2003)_5.png").exists()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        co2x0.vis_gt(src_path="no.png",dst_dir=Path(temp_dir))
        assert not Path(temp_dir).joinpath("a new precession formula(fukushima 2003)_5.png").exists()
        
    # 带有子目录的图片
    coco_path1 = TEST_DATA_PATH / "test02/annotations/instances_default.json"
    co2x1 = COCOX(coco_path1)
    with tempfile.TemporaryDirectory() as temp_dir:
        co2x1.vis_gt(src_path="a new precession formula(fukushima 2003)_5.png",dst_dir=Path(temp_dir))
        assert Path(temp_dir).joinpath("a new precession formula(fukushima 2003)_5.png").exists()
    
@pytest.fixture
def cc_merge11():
    ccx = CCX(ROOT=Path(TEST_DATA_PATH / "merge/to_merge11"),IMGFOLDER=Path("Validation"),ANNFILE=Path("instances_Validation.json"))
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge11/annotations/instances_Validation.json", cfg=ccx)

@pytest.fixture
def cc_merge12():
    ccx = CCX(ROOT=Path(TEST_DATA_PATH / "merge/to_merge12"),IMGFOLDER=Path("Train"),ANNFILE=Path("instances_Train.json"))
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge12/annotations/instances_Train.json",cfg=ccx)

@pytest.fixture
def cc_merge21():
    ccx = CCX(ROOT=Path(TEST_DATA_PATH / "merge/to_merge21"),IMGFOLDER=Path("Validation"),ANNFILE=Path("instances_Validation.json"))
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge21/annotations/instances_Validation.json", cfg=ccx)

@pytest.fixture
def cc_merge22():
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge22/annotations/instances_Train.json")
@pytest.fixture
def cc_merge23():
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge23/annotations/instances_Train.json")
@pytest.fixture
def cc_merge24():
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge24/annotations/instances_Train.json")
@pytest.fixture
def cc_merge25():
    ccx = CCX(ROOT=Path(TEST_DATA_PATH / "merge/to_merge24"),IMGFOLDER=Path("Train"),ANNFILE=Path("instances_Train1.json"))
    return COCOX(data=TEST_DATA_PATH / "merge/to_merge24/annotations/instances_Train1.json",cfg=ccx)

def test_merge_coco1(cc_merge11,cc_merge12):
    # 测试合并两个数据集
    # 均包含子目录
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge1 = CCX(ROOT=Path(temp_dir),IMGFOLDER=Path("Merge"),ANNFILE=Path("instances_Merge.json"))
        cc_merge1 = COCOX(cfg=ccx_merge1)
        cc_merge1 = cc_merge1.merge(others=cc_merge11,cat_keep=True,overwrite=True,dst_file=ccx_merge1,save_img=True)
        cc_merge1 = cc_merge1.merge(others=cc_merge12,cat_keep=True,overwrite=True,dst_file=ccx_merge1,save_img=True)
        
        # 判断数量
        assert len(cc_merge11.data.anns) + len(cc_merge12.data.anns) == len(cc_merge1.data.anns)
        assert len(cc_merge11.data.imgs) + len(cc_merge12.data.imgs) == len(cc_merge1.data.imgs)
        print(temp_dir)
        # 判断合并图片数量
        assert len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.jpg"))) + \
               len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.png"))) + \
               len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.jpeg"))) == len(cc_merge1.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge1.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()
    
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge1 = CCX(ROOT=Path(temp_dir),IMGFOLDER=Path("Merge"),ANNFILE=Path("instances_Merge.json"))
        cc_merge1 = COCOX(cfg=ccx_merge1)
        # 另一种合并方法
        cc_merge1 = cc_merge1.merge(others=[cc_merge11,cc_merge12],cat_keep=True,overwrite=True,dst_file=ccx_merge1,save_img=True)
        
        # 判断数量
        assert len(cc_merge11.data.anns) + len(cc_merge12.data.anns) == len(cc_merge1.data.anns)
        assert len(cc_merge11.data.imgs) + len(cc_merge12.data.imgs) == len(cc_merge1.data.imgs)
        print(temp_dir)
        # 判断合并图片数量
        assert len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.jpg"))) + \
               len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.png"))) + \
               len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.jpeg"))) == len(cc_merge1.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge1.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()
    

    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge1 = CCX(ROOT=Path(temp_dir),IMGFOLDER=Path("Merge"),ANNFILE=Path("instances_Merge.json"))
        cc_merge1 = COCOX(cfg=ccx_merge1)
        # 默认参数合并
        cc_merge1 = cc_merge1.merge(others=[cc_merge11,cc_merge12])
        
        # 判断数量
        assert len(cc_merge11.data.anns) + len(cc_merge12.data.anns) == len(cc_merge1.data.anns)
        assert len(cc_merge11.data.imgs) + len(cc_merge12.data.imgs) == len(cc_merge1.data.imgs)
        print(temp_dir)
        # 判断合并图片数量
        assert len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.jpg"))) + \
               len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.png"))) + \
               len(list(Path(temp_dir).joinpath("images").joinpath("Merge").glob("*.jpeg"))) == len(cc_merge1.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge1.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()
    
        
        
def test_merge_coco2(cc_merge21,cc_merge22):
    # 一个包含子目录一个不含子目录
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge2 = CCX(ROOT=Path(temp_dir),ANNFILE=Path("instances_Merge.json"))
        cc_merge2 = COCOX(cfg=ccx_merge2)
        cc_merge2 = cc_merge2.merge(others=cc_merge21,cat_keep=True,overwrite=True,dst_file=ccx_merge2,save_img=True)
        cc_merge2 = cc_merge2.merge(others=cc_merge22,cat_keep=True,overwrite=True,dst_file=ccx_merge2,save_img=True)
        
        # 判断数量
        assert len(cc_merge21.data.anns) + len(cc_merge22.data.anns) == len(cc_merge2.data.anns)
        assert len(cc_merge21.data.imgs) + len(cc_merge22.data.imgs) == len(cc_merge2.data.imgs)    
        # 测试合并多个数据集,默认值为.
        assert len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpg"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.png"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpeg"))) == len(cc_merge2.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge2.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()
    
    
     
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge2 = CCX(ROOT=Path(temp_dir),ANNFILE=Path("instances_Merge.json"))
        cc_merge2 = COCOX(cfg=ccx_merge2)
        # 另一种合并方法
        cc_merge2 = cc_merge2.merge(others=[cc_merge21,cc_merge22],cat_keep=True,overwrite=True,dst_file=ccx_merge2,save_img=True)
        
        # 判断数量
        assert len(cc_merge21.data.anns) + len(cc_merge22.data.anns) == len(cc_merge2.data.anns)
        assert len(cc_merge21.data.imgs) + len(cc_merge22.data.imgs) == len(cc_merge2.data.imgs)    
        # 测试合并多个数据集,默认值为.
        assert len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpg"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.png"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpeg"))) == len(cc_merge2.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge2.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()
        
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge2 = CCX(ROOT=Path(temp_dir),ANNFILE=Path("instances_Merge.json"))
        cc_merge2 = COCOX(cfg=ccx_merge2)
        # 默认参数合并
        cc_merge2 = cc_merge2.merge(others=[cc_merge21,cc_merge22])
        
        # 判断数量
        assert len(cc_merge21.data.anns) + len(cc_merge22.data.anns) == len(cc_merge2.data.anns)
        assert len(cc_merge21.data.imgs) + len(cc_merge22.data.imgs) == len(cc_merge2.data.imgs)    
        # 测试合并多个数据集,默认值为.
        assert len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpg"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.png"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpeg"))) == len(cc_merge2.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge2.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()


       
def test_merge_coco3(cc_merge21,cc_merge22, cc_merge23):
    # 三个数据集合并
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge2 = CCX(ROOT=Path(temp_dir),ANNFILE=Path("instances_Merge.json"))
        cc_merge2 = COCOX(cfg=ccx_merge2)
        cc_merge2 = cc_merge2.merge(others=[cc_merge21,cc_merge22,cc_merge23],cat_keep=True,overwrite=True,dst_file=ccx_merge2,save_img=True)
        
        
        # 判断数量
        assert len(cc_merge21.data.anns) + len(cc_merge22.data.anns) + len(cc_merge23.data.anns) == len(cc_merge2.data.anns)
        assert len(cc_merge21.data.imgs) + len(cc_merge22.data.imgs) + len(cc_merge23.data.imgs) == len(cc_merge2.data.imgs)    
        # 测试合并多个数据集,默认值为.
        assert len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpg"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.png"))) + \
                len(list(Path(temp_dir).joinpath("images").joinpath(ccx_merge2.IMGFOLDER).glob("*.jpeg"))) == len(cc_merge2.data.imgs)
        
        # 标注保存，因为图片在合并阶段必须保存，所以标注保存时不需要保存图片
        cc_merge2.save_data(visual=False,yolo=False,only_ann=True,overwrite=True)
        
        assert Path(temp_dir).joinpath("annotations").joinpath("instances_Merge.json").exists()

def test_merge_coco4(cc_merge21,cc_merge24):
    # 两个数据合并，类别序号错乱，但类别名称相同，大小写也需要一致
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx_merge4 = CCX(ROOT=Path(temp_dir),ANNFILE=Path("instances_Merge.json"))
        cc_merge4 = COCOX(cfg=ccx_merge4)
        cc_merge4 = cc_merge4.merge(others=[cc_merge21,cc_merge24],cat_keep=True,overwrite=True,dst_file=ccx_merge4,save_img=True)
        
        assert cc_merge4._get_map_cats() == cc_merge21._get_map_cats()

def test_rename_cat(sample_cc):
    """测试重命名类别"""
    sample_cc.rename_cat(raw_cat='class3',new_cat='class4')
    assert sample_cc._get_map_cats() == {1: 'class1', 2: 'class2', 3: 'class4'}



def test_update_cat(cc_merge21):
    """测试更新类别,仅针对相同的类别，否则会有问题"""
    # 乱序
    newCat = {3:'Text',2:'Table',1:'Formula',4:'Figure'}
    cc_merge21.update_cat(new_cat=newCat) # 1 Text 2 Head 3 Formula 4 Table
    assert {cat['id']: cat['name'] for cat in cc_merge21.data.dataset['categories']} == newCat 
    
def test_align_cat(cc_merge21):
    """TODO:测试对齐类别"""
    pass
    # # 乱序
    # newCat = {3:'Text',2:'Table',1:'Formula',4:'Figure'}
    # cc_merge21.align_cat(other_cat=newCat,cat_keep=True) # 1 Text 2 Head 3 Formula 4 Table
    # assert {cat['id']: cat['name'] for cat in cc_merge21.data.dataset['categories']} == newCat 
    
@pytest.fixture
def cc_filter():
    return COCOX(data=TEST_DATA_PATH / "filter/demo01/annotations/instances_Train.json")

def test_filter_and(cc_filter):
    """筛选同时满足类别、图片、标注"""
    result = cc_filter._filter(imgIds=[2],catIds=[1],annIds=[1])
    assert len(result) == 1
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)

def test_filter_or(cc_filter):
    """筛选满足其一的类别、图片、标注"""
    result = cc_filter._filter(catIds=[1],annIds=[2],mod="or")
    assert len(result) == 2
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
    # 验证结果包含所有catId=1或imgId=1的标注
    cat_anns = cc_filter._filter_and(catIds=[1])
    img_anns = cc_filter._filter_and(imgIds=[1])
    assert all(ann in result for ann in cat_anns)
    assert all(ann in result for ann in img_anns)

def test_filter_alignCat(cc_filter):
    """测试filter函数的alignCat=True参数"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        result = cc_filter.filter(imgs=[2,], dst_file=dst_file, alignCat=True)
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['categories']) == len(cc_filter.data.dataset['categories'])
        assert all(cat in result.data.dataset['categories'] for cat in cc_filter.data.dataset['categories'])

    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        result = cc_filter.filter(imgs=[2,], dst_file=dst_file, alignCat=False)
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['categories']) == len(cc_filter.data.dataset['categories'])
        assert all(cat in result.data.dataset['categories'] for cat in cc_filter.data.dataset['categories'])
        
def test_filter_multi_cond(cc_filter):
    """测试filter函数的多条件and模式"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        result = cc_filter.filter(imgs=["table3.png"], dst_file=dst_file, mod="and")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1
        assert len(result.data.dataset['annotations']) == 3

def test_filter_mohu_find(cc_filter):
    """测试filter函数,模糊搜索但仅支持单个结果"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        result = cc_filter.filter(imgs=["table"], dst_file=dst_file, mod="or")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1
        assert len(result.data.dataset['annotations']) == 3

def test_filter_multi_cond_or(cc_filter):
    """测试filter函数的多条件or模式"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        result = cc_filter.filter(imgs=["table4.jpg"],cats=["Text"], dst_file=dst_file, mod="or")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 2
        assert len(result.data.dataset['annotations']) == 3
        
def test_filter_multi_cond_level_ann(cc_filter):
    """测试filter函数的ann级别过滤"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        result = cc_filter.filter(imgs=["table3.png"],cats=["Text"], dst_file=dst_file, mod="and", level="ann")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1
        assert len(result.data.dataset['annotations']) == 1
        # 验证标注级别的过滤结果
        ann = result.data.dataset['annotations'][0]
        cat = next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])
        img = next(i for i in result.data.dataset['images'] if i['id'] == ann['image_id'])
        assert cat['name'] == "Text"
        assert "table3.png" in img['file_name']
        
def test_filter_sep_level_img(cc_filter):
    import os
    """测试filter函数的img级别过滤和数据分离"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        src_nums = len(cc_filter.data.dataset['images'])
        src_imgs = len(os.listdir(os.path.join(cc_filter.cfg.ROOT,"images")))
        
        result = cc_filter.filter(imgs=["table3.png"],cats=["Text"], dst_file=dst_file, level="img")
        result.save_data()
        filter_nums = len(result.data.dataset['images'])
        filter_imgs = len(os.listdir(os.path.join(result.cfg.ROOT,"images")))
        assert filter_nums == filter_imgs == 1
        assert len(cc_filter.data.dataset['annotations']) == 3
        
        # 验证原数据集保持不变
        assert len(cc_filter.data.dataset['images']) == src_nums
        assert len(os.listdir(os.path.join(cc_filter.cfg.ROOT,"images"))) == src_imgs
        
        # 验证过滤后的图片都包含目标类别或文件名
        for img in result.data.dataset['images']:
            assert "table3.png" in img['file_name']
            img_anns = [ann for ann in result.data.dataset['annotations'] if ann['image_id'] == img['id']]
            assert any(next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])['name'] == "Text" 
                      for ann in img_anns)
        
        # 测试保存功能
        with tempfile.TemporaryDirectory() as temp_dir2:
            new_dst = CCX(ROOT=Path(temp_dir2))
            result.save_data(dst_file=new_dst)
            assert Path(temp_dir2).exists()
            assert len(os.listdir(os.path.join(temp_dir2, "images"))) == filter_nums

# 存在问题
def test_filter_sep_level_ann(cc_filter):
    import os
    """测试filter函数的img级别过滤和数据分离"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCX(ROOT=Path(temp_dir))
        src_nums = len(cc_filter.data.dataset['images'])
        src_imgs = len(os.listdir(os.path.join(cc_filter.cfg.ROOT,"images")))
        
        result = cc_filter.filter(imgs=["table3.png"],cats=["Text"], dst_file=dst_file, level="ann")
        result.save_data()
        filter_nums = len(result.data.dataset['images'])
        filter_imgs = len(os.listdir(os.path.join(result.cfg.ROOT,"images")))
        assert filter_nums == filter_imgs == 1
        assert len(result.data.anns) == 1
        
        # 验证原数据集保持不变
        assert len(cc_filter.data.imgs) == src_nums
        assert len(os.listdir(os.path.join(cc_filter.cfg.ROOT,"images"))) == src_imgs
        
        # 验证过滤后的图片都包含目标类别或文件名
        for img in result.data.dataset['images']:
            assert "table3.png" in img['file_name']
            img_anns = [ann for ann in result.data.dataset['annotations'] if ann['image_id'] == img['id']]
            assert any(next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])['name'] == "Text" 
                      for ann in img_anns)
        
        # 测试保存功能
        with tempfile.TemporaryDirectory() as temp_dir2:
            new_dst = CCX(ROOT=Path(temp_dir2))
            result.save_data(dst_file=new_dst)
            assert Path(temp_dir2).exists()
            assert len(os.listdir(os.path.join(temp_dir2, "images"))) == filter_nums
    

def test_get_imgIds_by_annIds(cc_filter):
    """测试_get_imgIds_by_annIds函数"""
    result = cc_filter._get_imgIds_by_annIds(annIds=[2])
    assert result == [2]
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)        


def test_re_index(cc_filter):
    """测试re_index函数"""
    cc_filter.re_index()
    assert cc_filter.data.dataset['images'][0]['id'] == 1
    assert cc_filter.data.dataset['annotations'][0]['image_id'] == 1
    assert cc_filter.data.dataset['categories'][0]['id'] == 1


@pytest.fixture
def ccx_split():
    return COCOX(data=TEST_DATA_PATH / "coco_data/annotations/instances_default.json")


def test_split(ccx_split):
    import os
    # 划分为三部分，合并文件夹，即images不再存在子文件夹
    ratio = [0.7,0.2,0.1]
    total_nums = 50
    result:dict = ccx_split.split(ratio=ratio,merge=True) #35,10,5
    
    img_nums = [int(total_nums*r) for r in ratio]
    ann_nums = {k: 0 for k in result.keys()}
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx = CCX(ROOT=Path(temp_dir),IMGFOLDER=".")
        for i, (name, c2x) in enumerate(result.items()):
            assert len(c2x.data.dataset['images']) == img_nums[i]
            ccx.ANNFILE = f"instances_{name}.json"
            c2x = c2x.save_data(dst_file=ccx)
            # 验证保存的图片文件夹和标注文件是否存在
            assert c2x.cfg.ROOT.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER).exists()
            assert c2x.cfg.ROOT.joinpath(ccx.ANNDIR).joinpath(ccx.ANNFILE).exists()
            ann_nums[name] = len(c2x.data.anns)
        # 验证合并的图片文件夹图片数量
        assert len(os.listdir(c2x.cfg.ROOT.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER))) == total_nums

        assert len(os.listdir(c2x.cfg.ROOT.joinpath(ccx.IMGDIR))) == len(os.listdir(c2x.cfg.ROOT.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER)))
    
    assert sum(ann_nums.values()) == sum(len(c2x.data.anns) for c2x in result.values())
    
    
    # 划分为四部分，并自定义名称,不合并文件夹
    ratio = [0.5,0.2,0.1,0.2]
    ratio_name = ["data1","data2","data3","data4"]
    total_nums = 50
    result:dict = ccx_split.split(ratio=ratio,ratio_name=ratio_name) #25,10,5,10
    
    img_nums = [int(total_nums*r) for r in ratio]
    ann_nums = {k: 0 for k in result.keys()}
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (name, c2x) in enumerate(result.items()):
            assert len(c2x.data.dataset['images']) == img_nums[i]
            ccx = CCX(ROOT=Path(temp_dir),IMGFOLDER=name,ANNFILE=f"instances_{name}.json")
            c2x = c2x.save_data(dst_file=ccx)
            # 验证保存的图片文件夹和标注文件是否存在
            assert c2x.cfg.ROOT.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER).exists()
            assert c2x.cfg.ROOT.joinpath(ccx.ANNDIR).joinpath(ccx.ANNFILE).exists()
            # 验证单独文件夹保存的图片数量
            assert len(os.listdir(c2x.cfg.ROOT.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER))) == img_nums[i]
            ann_nums[name] = len(c2x.data.anns)
        # 不合并文件夹，则会有多个文件夹
        assert set(os.listdir(c2x.cfg.ROOT.joinpath(ccx.IMGDIR))) == set(ratio_name)
        assert len(os.listdir(c2x.cfg.ROOT.joinpath(ccx.IMGDIR))) == len(ratio_name)
    
    assert sum(ann_nums.values()) == sum(len(c2x.data.anns) for c2x in result.values())
    
def test_filter_save_yolo(ccx_split):
    result:dict = ccx_split.split(ratio=[0.9,0.1]) #25,10,5,10
    with tempfile.TemporaryDirectory() as temp_dir:
        ccx = CCX(ROOT=Path(temp_dir))
        c2x_val = result['val']
        c2x_val.save_data(dst_file=ccx,yolo=True)
        assert ccx.ROOT.joinpath(ccx.YOLODIR).exists()
        assert len(list(ccx.ROOT.joinpath(ccx.YOLODIR).glob("*.jpg"))) == \
            len(list(ccx.ROOT.joinpath(ccx.YOLODIR).glob("*.txt")))
        
    
    
def test_filter(ccx_split):
    """测试filter函数"""
    result = ccx_split.filter(imgs=["000000163611.jpg","000000088250.jpg"])
    assert len(result.data.dataset['images']) == 1+1
    assert len(result.data.dataset['annotations']) == 6+2
    
    # 不存在数据
    result = ccx_split.filter(imgs=["000000163611.jpg","no.jpg"])
    assert len(result.data.dataset['images']) == 1+0
    assert len(result.data.dataset['annotations']) == 6
    
    # 特定图片的特定类别，保留所有ann
    result = ccx_split.filter(imgs=["000000163611.jpg","000000088250.jpg"],cats=[59, 22])
    assert len(result.data.dataset['images']) == 1+1
    assert len(result.data.dataset['annotations']) == 6+2
    
    # 特定图片的特定类别，并且只过滤ann
    result = ccx_split.filter(imgs=["000000163611.jpg","000000088250.jpg"],cats=[59,22],level="ann")
    assert len(result.data.dataset['images']) == 1+1
    assert len(result.data.dataset['annotations']) == 2+2
    
    # 特定图片的特定类别，并且只过滤ann，去除无标注的图片
    result = ccx_split.filter(imgs=["000000163611.jpg","000000088250.jpg"],cats=[59,22],level="ann",keep_empty_img=False)
    assert len(result.data.dataset['images']) == 1+1
    assert len(result.data.dataset['annotations']) == 2+2
           
def test_re_index(ccx_split):
    """测试re_index函数"""
    ccx_split.re_index()
    assert ccx_split.data.dataset['images'][0]['id'] == 1
    assert ccx_split.data.dataset['annotations'][0]['image_id'] == 1
    assert ccx_split.data.dataset['categories'][0]['id'] == 1     
    

def test_correct_img_size(common_cc):
    """测试correct函数"""
    import os
    from pathlib import Path
    from cocox.utils.callback import callback_process_img_size
    from cocox.utils.common import IMG_EXT
    from cocox.cocox import CCX
    
    # 准备测试数据
    img_dir = TEST_DATA_PATH / "common/resize_images"
    dst_img_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(tuple(IMG_EXT)):
                dst_img_list.append(str(Path(root) / file))
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # 创建目标CCX对象
        ccx = CCX(ROOT=temp_dir_path)
        
        # 执行correct函数，不传入直接修改到common_cc下
        file_mode = 'copy'
        result = common_cc.correct(callback=callback_process_img_size, 
                                   img_list=dst_img_list, 
                                   dst_file=ccx, 
                                   file_mode=file_mode)
        # 全部纠正完才保存，避免数据被改动
        if len(dst_img_list) == 0:
            result.save_data()
            assert len(dst_img_list) == 0
        
        # 验证纠正后的图片是否存在
        assert temp_dir_path.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER).exists()
        # 使用list()将生成器转换为列表，然后再计算长度
        assert len(list(temp_dir_path.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER).iterdir())) == \
            len(list(common_cc.cfg.ROOT.joinpath(common_cc.cfg.IMGDIR).joinpath(common_cc.cfg.IMGFOLDER).iterdir()))
        # 验证纠正后的标注文件是否存在
        result.save_data()
        assert temp_dir_path.joinpath(ccx.ANNDIR).joinpath(ccx.ANNFILE).exists()

    
def test_correct_img_name(common_cc):
    """测试correct函数"""
    import copy
    from pathlib import Path
    from cocox.utils.callback import callback_process_img_name
    from cocox.cocox import CCX
        
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
    
        # 创建目标CCX对象
        ccx = CCX(ROOT=temp_dir_path)
        
        # 存在文件，并修改
        src_imgs=['a new precession formula(fukushima 2003)_5.png']
        dst_imgs=['a new precession formula(fukushima 2003)_5555.png']
        temp_imgs = copy.deepcopy(dst_imgs)
        result = common_cc.correct(callback=callback_process_img_name, dst_file=ccx,src_imgs=src_imgs,dst_imgs=dst_imgs)
        # 保留模式，两个图片同时存在，但标注已经修改了
        assert result.cfg.ROOT.joinpath(ccx.IMGDIR).joinpath(ccx.IMGFOLDER).joinpath(temp_imgs[0]).exists()
        # 验证标注文件也存在
        assert not result.cfg.ROOT.joinpath(ccx.ANNDIR).joinpath(ccx.ANNFILE).exists()
        assert len(src_imgs) == len(dst_imgs)  == 0
        
        src_imgs=['a new precession formula(fukushima 2003)_0.png']
        dst_imgs=['a new precession formula(fukushima 2003)_000.png']
        result = common_cc.correct(callback=callback_process_img_name, dst_file=ccx,src_imgs=src_imgs,dst_imgs=dst_imgs)
        # 不该处理
        assert len(src_imgs) == len(dst_imgs)  == 1


def test_big_merge():
    roots = [TEST_DATA_PATH / i for i in ["merge/to_merge11","merge/to_merge12"]]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_ccx = CCX(ROOT=Path(temp_dir))
        temp_c2x = COCOX(cfg=temp_ccx)
        
        # 对象为要保存的对象
        result = temp_c2x.big_merge(roots=roots)
        assert len(result.data.dataset['images']) == 3
        assert len(list(Path(temp_dir).joinpath(temp_ccx.IMGDIR).joinpath(temp_ccx.IMGFOLDER).glob("*.*"))) == 3
    
