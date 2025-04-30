from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

@dataclass
class CCX:
    """
    COCO数据集配置类
    
    属性:
        ROOT: 数据集根目录
        ANNDIR: 标注文件目录
        ANNFILE: 标注文件名
        IMGDIR: 图片目录
        IMGDIR_SRC: 图片源目录，默认为None，会自动设置为ROOT/IMGDIR
        
        IMGDIR_TRAIN: 训练集图片目录
        IMGDIR_VAL: 验证集图片目录
        IMGDIR_TEST: 测试集图片目录
        ANNFILE_TRAIN: 训练集标注文件名
        ANNFILE_VAL: 验证集标注文件名
        ANNFILE_TEST: 测试集标注文件名
        
        IMGDIR_VISUAL: 可视化结果保存目录
        YOLODIR: YOLO格式数据保存目录
    """
    ROOT: Optional[Path] = Path(".")
    ANNDIR: Optional[Path] = Path("annotations")
    ANNFILE: Optional[Path] = Path("instances_default.json")
    IMGDIR: Optional[Path] = Path("images") # 图片根目录
    IMGFOLDER: Optional[Path] = Path("default") # 默认图片目录
    IMGDIR_SRC: Optional[Path] = None  #完整路径，而不是单个文件夹名
    
    
    # delete
    IMGDIR_TRAIN: Optional[Path] = Path("train")
    IMGDIR_VAL: Optional[Path] = Path("val")
    IMGDIR_TEST: Optional[Path] = Path("test")
    ANNFILE_TRAIN: Optional[Path] = Path("instances_train.json")
    ANNFILE_VAL: Optional[Path] = Path("instances_val.json")
    ANNFILE_TEST: Optional[Path] = Path("instances_test.json")
    
    IMGDIR_VISUAL: Optional[Path] = Path("vis_images")
    IMGDIR_ANNFILE_VISUAL: Optional[Path] = Path("vis_feature")
    YOLODIR: Optional[Path] = Path("yolo")

# @dataclass
class STATIC_DATA:
    """
    数据集统计信息类
    
    属性: (dir表示图片文件夹目录，ann表示标注,img表示图片,annimg表示标注的图片)
        cats: dict{int: str} 类别统计 
        nums_cats: int 类别总数
        nums_anns: int 标注总数 
        nums_image_in_dir: int 目录中存在的图片数量
        nums_image_in_img: int 图片存在的图片数量
        nums_image_in_ann: int 标注存在的图片数量
        
        nums_img_ann_diff: int 图片和标注的图片不匹配的图片数量
        nums_img_dir_diff: int 图片和图片文件夹不匹配的图片数量
        
        list_img_ann_diff_in_img: list 图片和标注的图片不匹配的图片列表
        list_img_ann_diff_in_ann: list 标注和图片的图片不匹配的图片列表
        
        list_img_dir_diff_in_img: list 图片和图片文件夹不匹配的图片列表
        list_img_dir_diff_in_ann: list 图片文件夹和图片不匹配的图片列表
        
        # 一下图片均为ann提到的图片
        relate_cat_box: dict{str: int} 每个类别在图片中的标注数量统计 {类别名: 标注数量}
        relate_cat_img: dict{str: int} 每个类别对应的数量统计 {类别名: 图片数量}
        
    """
    cats: Dict[int, str] = field(default_factory=dict)
    nums_cats: int = 0
    nums_anns: int = 0
    nums_image_in_dir: int = 0
    nums_image_in_img: int = 0
    nums_image_in_ann: int = 0
    
    nums_img_ann_diff: int = 0
    nums_img_dir_diff: int = 0
    
    list_img_ann_diff_in_img: List[str] = []
    list_img_ann_diff_in_ann: List[str] = []
    
    list_img_dir_diff_in_img: List[str] = []
    list_img_dir_diff_in_ann: List[str] = []
    
    relate_cat_box: Dict[str, int] = {}
    relate_cat_img: Dict[str, List[str]] = {}
            

class Colors:
    """
    颜色类
    """
    def __init__(self):
        """初始化颜色列表"""
        # 十六进制颜色代码
        hexs = (
            "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", 
            "00FFFF", "FF8000", "8000FF", "008080", "808000", 
            "FF0080", "00FF80", "800000", "0080FF", "FFC0CC",
        )
        # 转换为RGB浮点值(0-1范围)
        self.colors_float = [self.hex2rgb_float(f"#{c}") for c in hexs]
        # 转换为RGB整数值(0-255范围)
        self.colors_int = [self.hex2rgb_int(f"#{c}") for c in hexs]
        # 保存原始十六进制值
        self.colors_hex = [f"#{c}" for c in hexs]
        self.n = len(self.colors_float)
    
    def __call__(self, i, bgr=False, format="float"):
        """
        返回颜色值
        
        参数:
            i: 颜色索引
            bgr: 是否返回BGR格式(用于OpenCV)
            format: 返回格式，可选值:
                - "float": 返回0-1范围的浮点值
                - "int": 返回0-255范围的整数值
                - "hex": 返回十六进制颜色代码
        """
        idx = int(i) % self.n
        
        if format == "float":
            c = self.colors_float[idx]
            return (c[2], c[1], c[0]) if bgr else c
        elif format == "int":
            c = self.colors_int[idx]
            return (c[2], c[1], c[0]) if bgr else c
        elif format == "hex":
            return self.colors_hex[idx]
        else:
            raise ValueError(f"不支持的格式: {format}，请使用 'float'、'int' 或 'hex'")
    
    @staticmethod
    def hex2rgb_float(h):
        """将十六进制颜色代码转换为RGB浮点值(0-1范围)"""
        r, g, b = tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return (r/255.0, g/255.0, b/255.0)
    
    @staticmethod
    def hex2rgb_int(h):
        """将十六进制颜色代码转换为RGB整数值(0-255范围)"""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    