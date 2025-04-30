import os,json, tempfile, pytest
from cctools import CCTools, CCFile, CC
from pathlib import Path
import copy
import numpy as np

# 在文件开头添加常量定义
TEST_ROOT = Path("tests/data/cctools")
TEST_IMGDIR = Path("images") 
TEST_ANNDIR = Path("annotations")

@pytest.fixture
def sample_coco_dict():
    return {
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

@pytest.fixture
def sample_coco_file():
    return CCTools(data="tests/data/cctools/annotations/instances_default.json")
        

@pytest.fixture 
def sample_cc(sample_coco_dict):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_coco_dict, f)
        temp_path = f.name
    return CC(temp_path)

# 添加新的fixtures
@pytest.fixture
def merge_one1():
    """第一个合并数据集"""
    return CCTools(data=TEST_ROOT / TEST_ANNDIR / "instances_one1.json")

@pytest.fixture
def merge_one2():
    """第二个合并数据集"""
    return CCTools(data=TEST_ROOT / TEST_ANNDIR / "instances_one2.json")


@pytest.fixture
def sample_split():
    """划分数据集"""
    return CCTools(data=TEST_ROOT / TEST_ANNDIR / "instances_split.json")


def test_init_with_dict(sample_coco_dict):
    """测试使用字典初始化"""
    cctools = CCTools(data=sample_coco_dict)
    assert cctools.data is not None
    assert len(cctools.data.dataset['images']) == 1
    assert len(cctools.data.dataset['annotations']) == 1
    assert len(cctools.data.dataset['categories']) == 1

def test_init_with_cc(sample_cc):
    """测试使用CC对象初始化"""
    cctools = CCTools(data=sample_cc)
    assert cctools.data is not None
    assert len(cctools.data.dataset['images']) == 1

def test_init_with_cctools(sample_coco_dict):
    """测试使用CCTools对象初始化"""
    cctools1 = CCTools(data=sample_coco_dict)
    cctools2 = CCTools(data=cctools1)
    assert cctools2.data is not None
    assert len(cctools2.data.dataset['images']) == 1
    assert cctools1.cfg == cctools2.cfg

def test_init_with_existing_path():
    """测试使用存在的路径初始化"""
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
            
        cctools = CCTools(data=coco_file)
        assert cctools.data is not None
        assert cctools.cfg.ROOT == temp_dir
        assert cctools.cfg.ANNDIR == Path("annotations")
        assert cctools.cfg.ANNFILE == Path("instances_default.json")

def test_init_with_nonexistent_path():
    """测试使用不存在的路径初始化"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir) / "newdir"
        cctools = CCTools(data=temp_dir)
        assert cctools.data is None
        assert cctools.cfg.ROOT == temp_dir
        assert temp_dir.exists()

def test_empty_init():
    """测试空初始化"""
    cctools = CCTools()
    assert cctools.data is None
    assert cctools.cfg.ROOT == Path(".")

def test_init_with_custom_cfg():
    """测试使用自定义cfg初始化"""
    custom_cfg = CCFile(
        ROOT=Path("/custom/root"),
        ANNDIR=Path("custom_annotations"),
        IMGDIR=Path("custom_images"),
        ANNFILE=Path("custom.json")
    )
    cctools = CCTools(cfg=custom_cfg)
    assert cctools.cfg.ROOT == Path("/custom/root")
    assert cctools.cfg.ANNDIR == Path("custom_annotations") 
    assert cctools.cfg.IMGDIR == Path("custom_images")
    assert cctools.cfg.ANNFILE == Path("custom.json")

def test_init_with_data_and_cfg(sample_coco_dict):
    """测试同时使用data和cfg初始化"""
    custom_cfg = CCFile(
        ROOT=Path("/custom/root"),
        ANNDIR=Path("custom_annotations")
    )
    cctools = CCTools(data=sample_coco_dict, cfg=custom_cfg)
    assert cctools.data is not None
    assert cctools.cfg.ROOT == Path("/custom/root")
    assert cctools.cfg.ANNDIR == Path("custom_annotations")
    assert len(cctools.data.dataset['images']) == 1

@pytest.fixture
def sample_coco_dict_with_empty_img():
    """创建包含空图片(没有标注)的数据集"""
    return {
        "images": [
            {"id": 1, "file_name": "test1.jpg", "height": 100, "width": 100},
            {"id": 2, "file_name": "test2.jpg", "height": 100, "width": 100}  # 空图片
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30]}
        ],
        "categories": [
            {"id": 1, "name": "test", "supercategory": ""}
        ]
    }

@pytest.fixture
def sample_image_dir():
    """创建临时图片目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        # 创建images目录
        img_dir = temp_dir / "images"
        img_dir.mkdir()
        # 创建测试图片文件
        (img_dir / "test1.jpg").touch()
        (img_dir / "test2.jpg").touch()
        (img_dir / "extra.jpg").touch()  # 额外的图片
        yield temp_dir

def test_init_with_validate_data(sample_coco_dict, sample_image_dir):
    """测试初始化时的数据验证"""
    cctools = CCTools(
        data=sample_coco_dict, 
        cfg=CCFile(
            ROOT=sample_image_dir,
            IMGDIR_SRC=sample_image_dir / "images"
        )
    )
    
    stats = cctools.static()
    assert stats["imgs"] == 1  # 标注文件中的图片数量
    assert stats["anns"] == 1  # 标注数量
    assert len(stats["cats"]) == 1  # 类别数量
    assert stats["img_in_folder"] == 3  # 文件夹中的图片数量

def test_init_with_empty_images(sample_coco_dict_with_empty_img):
    """测试包含空图片的数据集初始化"""
    cctools = CCTools(data=sample_coco_dict_with_empty_img)
    
    stats = cctools.static()
    assert stats["imgs"] == 2  # 总图片数量
    assert stats["anns"] == 1  # 标注数量
    assert stats["img_in_ann"] == 1  # 有标注的图片数量
    assert len(stats["img_in_ann_list"]) == 1  # 有标注的图片列表长度

def test_init_with_missing_images(sample_coco_dict, sample_image_dir):
    """测试缺失图片的数据集初始化"""
    sample_coco_dict["images"].append({
        "id": 2, 
        "file_name": "missing.jpg", 
        "height": 100, 
        "width": 100
    })
    
    cctools = CCTools(
        data=sample_coco_dict,
        cfg=CCFile(
            ROOT=sample_image_dir,
            IMGDIR_SRC=sample_image_dir / "images"
        )
    )
    
    stats = cctools.static()
    assert "missing.jpg" in stats["imgs_list"]  # 检查图片是否在列表中
    assert stats["imgs"] == 2  # 总图片数量
    assert stats["img_in_folder"] == 3  # 文件夹中的图片数量

def test_init_with_correct_data(sample_coco_dict, sample_image_dir):
    """测试使用correct_data参数初始化"""
    # 添加一个不存在的图片
    sample_coco_dict["images"].append({
        "id": 2, 
        "file_name": "test1.jpg", 
        "height": 100, 
        "width": 100
    })
    sample_coco_dict["annotations"].append({
        "id": 2,
        "image_id": 2,
        "category_id": 1,
        "bbox": [10, 10, 30, 30]
    })
    
    cctools = CCTools(data=sample_coco_dict,
                     cfg=CCFile(ROOT=sample_image_dir),
                     correct_data=True)
    
    # 检查数据是否被正确清理
    assert len(cctools.data.dataset["images"]) == 1
    assert len(cctools.data.dataset["annotations"]) == 1
    assert "missing.jpg" not in [img["file_name"] for img in cctools.data.dataset["images"]]

def test_static_with_empty_data():
    """测试空数据集的统计"""
    cctools = CCTools()
    stats = cctools.static()
    
    assert stats["imgs"] == 0
    assert stats["anns"] == 0
    assert stats["cats"] == {}
    assert stats["img_in_ann"] == 0
    assert len(stats.get("img_in_ann_list", [])) == 0

def test_update_cfg():
    """测试配置更新"""
    custom_cfg = CCFile(
        ROOT=Path("/custom/root"),
        ANNDIR=Path("custom_annotations"),
        IMGDIR=Path("custom_images"),
    )
    
    # 使用部分配置更新
    new_cfg = CCFile(
        ROOT=Path("/new/root"),
        ANNDIR=Path("new_annotations")
    )
    
    cctools = CCTools(cfg=custom_cfg)
    cctools._update_cfg(new_cfg)
    
    assert cctools.cfg.ROOT == Path("/new/root")
    assert cctools.cfg.ANNDIR == Path("new_annotations")
    assert cctools.cfg.IMGDIR == Path("custom_images")  # 保持不变
    assert cctools.cfg.ANNFILE == Path("instances_default.json")   # 保持不变

def test_save_yolo(sample_coco_dict, sample_image_dir):
    """测试YOLO格式数据保存"""
    # 准备测试数据
    sample_coco_dict["images"] = [{
        "id": 1,
        "file_name": "test1.jpg",
        "height": 100,
        "width": 100
    }]
    sample_coco_dict["annotations"] = [{
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10, 10, 30, 30]  # [x,y,w,h]格式
    }]
    
    # 初始化CCTools对象
    cctools = CCTools(
        data=sample_coco_dict,
        cfg=CCFile(
            ROOT=sample_image_dir,
            YOLODIR=Path("yolo")
        )
    )
    
   
    # 保存YOLO格式数据
    cctools.save_yolo()
    
    # 验证YOLO目录是否创建
    yolo_dir = sample_image_dir / "yolo"
    assert yolo_dir.exists()
    
    # 验证图片是否复制
    assert (yolo_dir / "test1.jpg").exists()
    
    # 验证标签文件是否生成
    label_file = yolo_dir / "test1.txt"
    assert label_file.exists()
    
    # 验证标签内容是否正确
    with open(label_file) as f:
        content = f.read().strip()
        # YOLO格式: <class> <x_center> <y_center> <width> <height>
        # 这里的数值应该是归一化后的结果
        values = [float(x) for x in content.split()]
        assert len(values) == 5  # 类别 + 4个坐标值
        assert values[0] == 0  # 类别索引(从0开始)
        assert 0 <= values[1] <= 1  # x_center
        assert 0 <= values[2] <= 1  # y_center
        assert 0 <= values[3] <= 1  # width
        assert 0 <= values[4] <= 1  # height

def test_save_yolo_multiple_classes(sample_image_dir):
    """测试多类别数据的YOLO格式保存"""
    # 准备包含多个类别的测试数据
    multi_class_data = {
        "images": [{
            "id": 1,
            "file_name": "test1.jpg",
            "height": 100,
            "width": 100
        }],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 30]
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [50, 50, 20, 20]
            }
        ],
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"}
        ]
    }
    
    cctools = CCTools(
        data=multi_class_data,
        cfg=CCFile(
            ROOT=sample_image_dir,
            YOLODIR=Path("yolo")
        )
    )
    
    # 转换并保存YOLO格式
    cctools.save_yolo()
    
    # 验证id2cls映射是否正确
    assert cctools.other_data['yolo']['id2cls'] == {0: 'class1', 1: 'class2'}
    
    # 验证标签文件内容
    label_file = sample_image_dir / "yolo" / "test1.txt"
    with open(label_file) as f:
        lines = f.readlines()
        assert len(lines) == 2  # 两个标注
        
        # 验证每行的格式
        for line in lines:
            values = [float(x) for x in line.strip().split()]
            assert len(values) == 5
            assert values[0] in [0, 1]  # 类别索引
            assert all(0 <= v <= 1 for v in values[1:])  # 归一化坐标

def test_save_yolo_missing_image(sample_image_dir):
    """测试当图片文件缺失时的YOLO格式保存"""
    test_data = {
        "images": [{
            "id": 1,
            "file_name": "missing.jpg",
            "height": 100,
            "width": 100
        }],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 30, 30]
        }],
        "categories": [
            {"id": 1, "name": "test"}
        ]
    }
    
    cctools = CCTools(
        data=test_data,
        cfg=CCFile(
            ROOT=sample_image_dir,
            YOLODIR=Path("yolo")
        )
    )
    
    # 转换并保存YOLO格式
    cctools.save_yolo()
    
    # 验证标签文件是否生成
    label_file = sample_image_dir / "yolo" / "missing.txt"
    assert label_file.exists()
    
    # 验证图片文件确实不存在
    assert not (sample_image_dir / "yolo" / "missing.jpg").exists()

def test_2dict(sample_coco_dict):
    """测试数据转换为字典格式"""
    cctools = CCTools(data=sample_coco_dict)
    cctools._2dict()
    
    # 验证转换后的数据结构
    assert 'dict' in cctools.other_data
    assert 'data' in cctools.other_data['dict']
    assert 'id2cls' in cctools.other_data['dict']
    
    # 验证数据内容
    dict_data = cctools.other_data['dict']['data']
    assert len(dict_data['images']) == 1
    assert len(dict_data['annotations']) == 1
    assert len(dict_data['categories']) == 1
    
    # 验证类别映射
    assert cctools.other_data['dict']['id2cls'] == {1: 'test'}

def test_save_img(sample_coco_dict):
    """测试图片复制功能"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        src_dir = temp_dir / "source"
        dst_dir = temp_dir / "destination"
        
        # 创建源图片
        src_dir.mkdir()
        test_img = src_dir / "test.jpg"
        test_img.touch()
        
        # 初始化CCTools对象
        cctools = CCTools(
            data=sample_coco_dict,
            cfg=CCFile(
                ROOT=dst_dir,
                IMGDIR=Path("images"),
                IMGDIR_SRC=src_dir
            )
        )
        
        # 测试复制功能
        cctools.save_img()
        
        # 验证目标目录中的图片
        assert (dst_dir / "images" / "test.jpg").exists()
        
        # 验证源路径更新
        assert cctools.cfg.IMGDIR_SRC == dst_dir / "images"

def test_save_img_missing_source():
    """测试源图片缺失情况"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        src_dir = temp_dir / "source"
        dst_dir = temp_dir / "destination"
        
        # 创建源目录但不创建图片文件
        src_dir.mkdir()
        
        cctools = CCTools(
            data={"images": [{"id": 1, "file_name": "missing.jpg"}], "annotations": [], "categories": []},
            cfg=CCFile(
                ROOT=dst_dir,
                IMGDIR=Path("images"),
                IMGDIR_SRC=src_dir
            )
        )
        
        # 行复制,应该只产生警告而不是错误
        cctools.save_img()
        
        # 验证目标目录已创建但图片未复制
        assert (dst_dir / "images").exists()
        assert not (dst_dir / "images" / "missing.jpg").exists()

def test_save_annfile(sample_coco_dict):
    """测试保存标注文件"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        ann_dir = temp_dir / "annotations"
        ann_dir.mkdir()
        
        cctools = CCTools(
            data=sample_coco_dict,
            cfg=CCFile(
                ROOT=temp_dir,
                ANNDIR=Path("annotations"),
                ANNFILE=Path("test.json")
            )
        )
        
        # 转换为字典格式并保存
        cctools.save_annfile()
        
        # 验证文件是否创建
        ann_file = ann_dir / "test.json"
        assert ann_file.exists()
        
        # 验证文件内容
        with open(ann_file) as f:
            saved_data = json.load(f)
            assert len(saved_data['images']) == 1
            assert len(saved_data['annotations']) == 1
            assert len(saved_data['categories']) == 1

def test_save_data(sample_coco_file):
    """测试完整的数据保存功能"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # 创建目标配置
        dst_cfg = CCFile(
            ROOT=temp_dir / "destination",
            ANNDIR=Path("annotations"),
            IMGDIR=Path("images"),
            ANNFILE=Path("test.json")
        )
        
        # 确保目标目录存在
        (temp_dir / "destination" / "annotations").mkdir(parents=True, exist_ok=True)
        
        # 测试保存功能
        dst_cctools = sample_coco_file.save_data(
            dst_file=dst_cfg,
            visual=True,
            yolo=True,
            overwrite=True
        )
        
        # 验证目标目录结构
        assert (dst_cfg.ROOT / "images" / "a new precession formula(fukushima 2003)_5.jpg").exists()
        assert (dst_cfg.ROOT / "annotations" / "test.json").exists()
        assert (dst_cfg.ROOT / "yolo").exists()
        assert (dst_cfg.ROOT / "visual").exists()
        
        # 验证返回的对象
        assert isinstance(dst_cctools, CCTools)
        assert dst_cctools.cfg.ROOT == dst_cfg.ROOT

def test_save_data_only_ann(sample_coco_dict):
    """测试仅保存标注文件"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        cctools = CCTools(
            data=sample_coco_dict,
            cfg=CCFile(ROOT=temp_dir)
        )
        
        # 测试仅保存标注
        cctools.save_data(only_ann=True)
        
        # 验证只有标注文件被创建
        assert (temp_dir / "annotations" / "instances_default.json").exists()
        assert not (temp_dir / "images").exists()
        assert not (temp_dir / "yolo").exists()
        assert not (temp_dir / "visual").exists()

def test_update_cat(sample_coco_file):
    """测试更新类别"""
    newCat = {3:'Text',2:'Table',1:'Formula',4:'Figure'}
    sample_coco_file.update_cat(new_cat=newCat)
    assert {cat['id']: cat['name'] for cat in sample_coco_file.data.dataset['categories']} == newCat 

def test_rename_cat(sample_coco_file):
    """测试重命名类别"""
    sample_coco_file.rename_cat(raw_cat='Formula',new_cat='Formula2')
    assert {cat['id']: cat['name'] for cat in sample_coco_file.data.dataset['categories']} == {1: 'Text', 2: 'Table', 3: 'Formula2', 4: 'Figure'}

def test_get_cat(sample_coco_file):
    """测试获取类别"""
    assert sample_coco_file._get_cat(cat='Figure') == (True,4)
    assert sample_coco_file._get_cat(cat=4) == (True,'Figure')
    assert sample_coco_file._get_cat(cat='Figure2') == (False,None)
    assert sample_coco_file._get_cat(cat=5) == (False,None)


def test_get_img(sample_coco_file):
    assert sample_coco_file._get_img(img='a new precession formula(fukushima 2003)_5.jpg') == (True,1)    
    assert sample_coco_file._get_img(img=1) == (True,'a new precession formula(fukushima 2003)_5.jpg')
    assert sample_coco_file._get_img(img='a new precession formula(fukushima 2003)_5') == (True,1)    
    assert sample_coco_file._get_img(img='a new precession formula(fukus') == (True,1)    
    
def test_get_imglist(sample_coco_file):
    result = sample_coco_file._get_imglist()
    assert len(result) == 2

def test_rename_cat_in_ann(sample_coco_file):
    """测试在标注中重命名类别"""
    sample_coco_file.rename_cat_in_ann(old_name='Formula',new_name='Text')
    assert all([ann['category_id'] != 3 for ann in sample_coco_file.data.dataset['annotations']])



# 添加merge相关的测试用例
def test_merge_basic(merge_one1, merge_one2):
    """测试基本的合并功能"""
    # 记录原始数据
    orig_imgs1 = len(merge_one1.data.dataset['images'])
    orig_anns1 = len(merge_one1.data.dataset['annotations'])
    orig_cats1 = len(merge_one1.data.dataset['categories'])
    
    orig_imgs2 = len(merge_one2.data.dataset['images'])
    orig_anns2 = len(merge_one2.data.dataset['annotations'])
    orig_cats2 = len(merge_one2.data.dataset['categories'])
    
    # 执行合并
    merge_one1._merge(merge_one2)
    
    # 验证合并结果
    assert len(merge_one1.data.dataset['images']) == orig_imgs1 + orig_imgs2
    assert len(merge_one1.data.dataset['annotations']) == orig_anns1 + orig_anns2
    # 类别数可能不等于简单相加，因为可能有重复类别
    assert len(merge_one1.data.dataset['categories']) >= max(orig_cats1, orig_cats2)

def test_merge_with_cat_keep(merge_one1, merge_one2):
    """测试保留原类别ID的合并"""
    # 记录原始类别映射
    orig_cats1 = {cat['id']: cat['name'] for cat in merge_one1.data.dataset['categories']}
    
    # 执行合并，保留merge_one1的类别ID
    merge_one1._merge(merge_one2, cat_keep=True)
    
    # 验证原始类别ID保持不变
    new_cats = {cat['id']: cat['name'] for cat in merge_one1.data.dataset['categories']}
    for cat_id, cat_name in orig_cats1.items():
        assert cat_id in new_cats
        assert new_cats[cat_id] == cat_name

def test_merge_with_overwrite(merge_one1, merge_one2):
    """测试��盖重复图片的合并"""
    # 获取一个在两个数据集中都存在的图片名称
    common_img = set(merge_one1._get_imglist()) & set(merge_one2._get_imglist())
    if common_img:
        common_img = list(common_img)[0]
        # 记录merge_one2中该图片的标注数量
        img_id2 = merge_one2._get_img(common_img)[1]
        anns_count2 = len(merge_one2.data.getAnnIds(imgIds=[img_id2]))
        
        # 执行合并，允许覆盖
        merge_one1._merge(merge_one2, overwrite=True)
        
        # 验证图片的标注被正确覆盖
        img_id1 = merge_one1._get_img(common_img)[1]
        anns_count1 = len(merge_one1.data.getAnnIds(imgIds=[img_id1]))
        assert anns_count1 == anns_count2

def test_merge_multiple(merge_one1, merge_one2):
    """测试合并多个数据集"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dst_cfg = CCFile(
            ROOT=temp_dir,
            ANNDIR=Path("annotations"),
            IMGDIR=Path("images")
        )
        
        # 执行多数据集合并
        result = merge_one1.merge(
            others=[merge_one2],
            dst_file=dst_cfg,
            cat_keep=True,
            overwrite=False
        )
        
        # 验证合并结果
        assert isinstance(result, CCTools)
        assert result.cfg.ROOT == temp_dir
        assert len(result.data.dataset['images']) >= len(merge_one1.data.dataset['images'])
        assert len(result.data.dataset['annotations']) >= len(merge_one1.data.dataset['annotations'])
        
        # 验证目录结构
        result.save_data()
        assert (temp_dir / "annotations").exists()
        assert (temp_dir / "images").exists()


def test_merge_multiple_without_dst_file(merge_one1, merge_one2):
    """测试合并多个数据集"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # 执行多数据集合并
        result = merge_one1.merge(
            others=[merge_one2],
            cat_keep=True,
            overwrite=False
        )
        
        # 验证合并结果
        assert isinstance(result, CCTools)
        assert len(result.data.dataset['images']) >= len(merge_one1.data.dataset['images'])
        assert len(result.data.dataset['annotations']) >= len(merge_one1.data.dataset['annotations'])
        

def test_merge_with_empty_target():
    """测试目标数据集为空的合并"""
    # 创建空的目标数据集
    empty_target = CCTools()
    
    # 创建源数据集
    source = CCTools(data=TEST_ROOT / TEST_ANNDIR / "instances_one1.json")
    
    # 执行合并
    result = empty_target.merge(
        others=source,
        cat_keep=True
    )
    
    # 验证合并结果
    assert len(result.data.dataset['images']) == len(source.data.dataset['images'])
    assert len(result.data.dataset['annotations']) == len(source.data.dataset['annotations'])
    assert len(result.data.dataset['categories']) == len(source.data.dataset['categories'])

def test_merge_with_save(merge_one1, merge_one2):
    """测试合并时保存数据"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dst_cfg = CCFile(
            ROOT=temp_dir,
            ANNDIR=Path("annotations"),
            IMGDIR=Path("images"),
            ANNFILE=Path("merged.json")
        )
        
        # 执行合并并保存
        result = merge_one1.merge(
            others=merge_one2,
            dst_file=dst_cfg,
            save_img=True
        )
        result.save_data()
        # 验证保存的文件
        assert (temp_dir / "annotations" / "merged.json").exists()
        assert (temp_dir / "images").exists()
        assert len(list((temp_dir / "images").iterdir())) > 0


def test_filter_and(sample_coco_file):
    """测试_filter函数的and模式"""
    result = sample_coco_file._filter(catIds=[1],imgIds=[1],annIds=[2])
    assert len(result) == 1
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)

def test_filter_or(sample_coco_file):
    """测试_filter函数的or模式"""
    result = sample_coco_file._filter(catIds=[1],imgIds=[1],mod="or")
    assert len(result) == 12
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
    # 验证结果包含所有catId=1或imgId=1的标注
    cat_anns = sample_coco_file._filter_and(catIds=[1])
    img_anns = sample_coco_file._filter_and(imgIds=[1])
    assert all(ann in result for ann in cat_anns)
    assert all(ann in result for ann in img_anns)

def test_filter(sample_coco_file):
    """测试_filter函数的基本功能"""
    result = sample_coco_file._filter(catIds=[1],imgIds=[1,2],annIds=[1,2])
    assert len(result) == 2
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
    assert all(x in [1,2] for x in result)

def test_filter_alignCat(sample_coco_file):
    """测试filter函数的alignCat=True参数"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=[1,], dst_file=dst_file, alignCat=True)
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['categories']) == len(sample_coco_file.data.dataset['categories'])
        assert all(cat in result.data.dataset['categories'] for cat in sample_coco_file.data.dataset['categories'])
        
def test_filter_no_alignCat(sample_coco_file):
    """测试filter函数的alignCat=False参数"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=[1,], dst_file=dst_file, alignCat=False)
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1
        # 验证只保留了相关类别
        used_cats = set(ann['category_id'] for ann in result.data.dataset['annotations'])
        assert len(result.data.dataset['categories']) >= len(used_cats)

def test_get_imgIds_by_annIds(sample_coco_file):
    """测试_get_imgIds_by_annIds函数"""
    result = sample_coco_file._get_imgIds_by_annIds(annIds=[13,14])
    assert result == [2]
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)

def test_filter_multi_cond(sample_coco_file):
    """测试filter函数的多条件and模式"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=["xx"],cats=["Formula"], dst_file=dst_file, mod="and")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1


def test_filter_multi_cond_or(sample_coco_file):
    """测试filter函数的多条件or模式"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=["xx"],cats=["Formula"], dst_file=dst_file, mod="or")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 2

def test_filter_multi_cond_level_ann(sample_coco_file):
    """测试filter函数的ann级别过滤"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=["xx"],cats=["Formula"], dst_file=dst_file, mod="and", level="ann")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1
        assert len(result.data.dataset['annotations']) == 1
        # 验证标注级别的过滤结果
        ann = result.data.dataset['annotations'][0]
        cat = next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])
        img = next(i for i in result.data.dataset['images'] if i['id'] == ann['image_id'])
        assert cat['name'] == "Formula"
        assert "xx" in img['file_name']
        
def test_filter_sep_level_img(sample_coco_file):
    """测试filter函数的img级别过滤和数据分离"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        src_nums = len(sample_coco_file.data.dataset['images'])
        src_imgs = len(os.listdir(os.path.join(sample_coco_file.cfg.ROOT,"images")))
        
        result = sample_coco_file.filter(imgs=["xx"],cats=["Formula"], dst_file=dst_file, level="img")
        result.save_data()
        filter_nums = len(result.data.dataset['images'])
        filter_imgs = len(os.listdir(os.path.join(result.cfg.ROOT,"images")))
        assert filter_nums == filter_imgs
        assert filter_nums == 1
        
        # 验证原数据集保持不变
        assert len(sample_coco_file.data.dataset['images']) == src_nums
        assert len(os.listdir(os.path.join(sample_coco_file.cfg.ROOT,"images"))) == src_imgs
        
        # 验证过滤后的图片都包含目标类别或文件名
        for img in result.data.dataset['images']:
            assert "xx" in img['file_name']
            img_anns = [ann for ann in result.data.dataset['annotations'] if ann['image_id'] == img['id']]
            assert any(next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])['name'] == "Formula" 
                      for ann in img_anns)
        
        # 测试保存功能
        with tempfile.TemporaryDirectory() as temp_dir2:
            new_dst = CCFile(ROOT=Path(temp_dir2))
            result.save_data(dst_file=new_dst)
            assert Path(temp_dir2).exists()
            assert len(os.listdir(os.path.join(temp_dir2, "images"))) == filter_nums
        
def test_filter_sep_level_ann(sample_coco_file):
    """测试filter函数的ann级别过滤和数据分离"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=["xx"],cats=["Formula"], dst_file=dst_file, level="ann")
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 1
        
        # 验证标注级别的过滤结果
        for ann in result.data.dataset['annotations']:
            cat = next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])
            img = next(i for i in result.data.dataset['images'] if i['id'] == ann['image_id'])
            assert cat['name'] == "Formula"
            assert "xx" in img['file_name']
    
        new_dst = CCFile(ROOT=Path(temp_dir))
        result.save_data(dst_file=new_dst)
        assert Path(temp_dir).exists()
        assert os.path.exists(os.path.join(temp_dir, "images"))
        assert os.path.exists(os.path.join(temp_dir, "annotations"))
        
def test_filter_sep_level_img_alignCat_or(sample_coco_file):
    """测试filter函数的img级别过滤、类别对齐和or模式"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        result = sample_coco_file.filter(imgs=["xx"],cats=["Formula"], dst_file=dst_file, mod="or", level="img", alignCat=True)
        assert result.cfg.ROOT == Path(temp_dir)
        assert len(result.data.dataset['images']) == 2
        # 验证类别对齐
        assert len(result.data.dataset['categories']) == len(sample_coco_file.data.dataset['categories'])
        # 验证or模式的过滤结果
        for img in result.data.dataset['images']:
            img_anns = [ann for ann in result.data.dataset['annotations'] if ann['image_id'] == img['id']]
            assert "xx" in img['file_name'] or any(
                next(c for c in result.data.dataset['categories'] if c['id'] == ann['category_id'])['name'] == "Formula"
                for ann in img_anns
            )
        
        # 测试保存功能
        with tempfile.TemporaryDirectory() as temp_dir2:
            new_dst = CCFile(ROOT=Path(temp_dir2))
            result.save_data(dst_file=new_dst)
            assert Path(temp_dir2).exists()
            assert len(os.listdir(os.path.join(temp_dir2, "images"))) == 2
            assert len(result.data.dataset['images']) == 2
            
            
            
            
def test_correct(sample_coco_file):
    """测试correct函数删除指定类别的标注"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        
        # 测试基本功能
        result = sample_coco_file.correct(api_url=lambda x,y:1, cats=["Formula"], dst_file=dst_file)
        
        # 检查是否还存在formula类别的标注
        formula_cat_id = result._get_cat("Formula", return_id=True)
        formula_anns = [ann for ann in result.data.dataset['annotations'] 
                       if ann['category_id'] == formula_cat_id]
        assert len(formula_anns) == 0
        
        # 测试多个类别
        result = sample_coco_file.correct(api_url=lambda x,y:1, cats=["Formula", "Text"], dst_file=dst_file)
        formula_anns = [ann for ann in result.data.dataset['annotations'] 
                       if ann['category_id'] in [result._get_cat(cat, return_id=True) for cat in ["Formula", "Text"]]]
        assert len(formula_anns) == 0
        
        # 测试空类别列表
        result = sample_coco_file.correct(api_url=lambda x,y:1, cats=[], dst_file=dst_file)
        assert len(result.data.dataset['annotations']) == len(sample_coco_file.data.dataset['annotations'])

def test_split(sample_split):
    """测试split函数的数据集划分"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dst_file = CCFile(ROOT=Path(temp_dir))
        
        # 测试基本的二分类划分
        train_obj, val_obj,_ = sample_split.split(
            ratio=[0.7, 0.3],
            dst_file=dst_file,
            by_file=False
        )
        total_images = len(train_obj.data.dataset['images']) + len(val_obj.data.dataset['images'])
        assert total_images == len(sample_split.data.dataset['images'])
        
        # 测试三分类划分
        train_obj, val_obj, test_obj = sample_split.split(
            ratio=[0.6, 0.2, 0.2],
            dst_file=dst_file,
            by_file=False
        )
        total_images = len(train_obj.data.dataset['images']) + \
                      len(val_obj.data.dataset['images']) + \
                      len(test_obj.data.dataset['images'])
        assert total_images == len(sample_split.data.dataset['images'])
        
        # 测试按文件划分
        train_obj, val_obj,_ = sample_split.split(
            ratio=[0.7, 0.3],
            dst_file=dst_file,
            by_file=True
        )
        assert len(train_obj.data.dataset['images']) > 0
        assert len(val_obj.data.dataset['images']) > 0

def test_merge_annotations_consistency(merge_one1, merge_one2):
    """测试合并后标注的完整性"""
    # 记录原始数据
    orig_ann_counts = {}
    for img in merge_one1.data.dataset['images']:
        img_anns = [ann for ann in merge_one1.data.dataset['annotations'] 
                   if ann['image_id'] == img['id']]
        orig_ann_counts[img['file_name']] = len(img_anns)
        
    # 执行合并
    merge_one1._merge(merge_one2)
    
    # 验证合并后的标注
    for img in merge_one1.data.dataset['images']:
        img_anns = [ann for ann in merge_one1.data.dataset['annotations'] 
                   if ann['image_id'] == img['id']]
        # 如果是原始数据集中的图片，确保标注数量不少于原始数量
        if img['file_name'] in orig_ann_counts:
            assert len(img_anns) >= orig_ann_counts[img['file_name']]
        # 确保每张图片都有标注
        assert len(img_anns) > 0, f"Image {img['file_name']} has no annotations"
