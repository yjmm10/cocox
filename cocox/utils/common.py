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
    ANNFILE: Optional[Path] = None # Path("instances_default.json")
    IMGDIR: Optional[Path] = Path("images") # 图片根目录
    IMGFOLDER: Optional[Path] = None # Path(".") # 默认图片目录
    IMGDIR_SRC: Optional[Path] = None  #完整路径，而不是单个文件夹名

    IMGDIR_VISUAL: Optional[Path] = Path("vis_images")
    IMGDIR_ANNFILE_VISUAL: Optional[Path] = Path("vis_feature")
    YOLODIR: Optional[Path] = Path("yolo")
    
    def __setattr__(self, name, value):
        """
        重写属性设置方法，确保所有路径类型的属性都被转换为Path对象。
        
        Args:
            name: 属性名
            value: 属性值
        """
        # 如果值不是None且不是Path类型，则尝试转换为Path对象
        if value is not None and not isinstance(value, Path):
            try:
                value = Path(value)
            except:
                pass
        # 调用父类的__setattr__方法设置属性
        super().__setattr__(name, value)
    
    def __post_init__(self):
        """
        初始化后处理函数，用于自动设置ANNFILE和IMGFOLDER的值。
        
        该函数在对象创建后自动调用，主要完成以下工作：
        1. 设置默认的标注文件名和图片文件夹名
        2. 根据ANNFILE和IMGFOLDER的存在情况进行互相转换：
           - 如果只有IMGFOLDER，则根据它生成ANNFILE
           - 如果只有ANNFILE，则从它提取IMGFOLDER
           - 如果都不存在，则使用默认值
        
        注意：
        - ANNFILE命名规则为"instances_{IMGFOLDER}.json"
        - 从ANNFILE提取IMGFOLDER时，会去除"instances_"前缀
        """

                
        # 如果IMGDIR_SRC为None，则自动设置为ROOT/IMGDIR
        temp_annfile = Path("instances_default.json")
        temp_imgfolder = Path(".")
        # ANNFILE与IMGFOLDER互相转化
        if self.ANNFILE is None and self.IMGFOLDER is not None:
            # 如果IMGFOLDER存在但ANNFILE不存在，则根据IMGFOLDER设置ANNFILE
            # 当imgfolder为.时，则设置为default
            if self.IMGFOLDER == Path("."):
                self.ANNFILE = Path("instances_default.json")
            else:
                self.ANNFILE = Path(f"instances_{self.IMGFOLDER}.json")
        elif self.ANNFILE is not None and self.IMGFOLDER is None:
            # 如果ANNFILE存在但IMGFOLDER不存在，则从ANNFILE获取IMGFOLDER
            annfile_stem = self.ANNFILE.stem
            if annfile_stem.startswith("instances_"):
                self.IMGFOLDER = Path(annfile_stem[len("instances_"):])
            else:
                self.IMGFOLDER = Path(annfile_stem)
        elif self.ANNFILE is None and self.IMGFOLDER is None:
            # 如果都不存在，则使用默认值
            self.ANNFILE = temp_annfile
            self.IMGFOLDER = temp_imgfolder
        # 将所有路径类型的属性转换为Path对象
        for attr_name, attr_value in self.__dict__.items():
            if attr_value is not None and not isinstance(attr_value, Path):
                setattr(self, attr_name, Path(attr_value))

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


IMG_EXT = ['.jpg','.png','.jpeg','.bmp','.tiff','.gif']




# 打包指定数据
def zip_data(data_dir, zip_file_path, file_exts=None):
    """
    打包指定数据目录中的特定文件格式
    
    参数:
        data_dir: 数据目录路径
        zip_file_path: 生成的zip文件保存路径,指定到 *.zip
        file_exts: 要包含的文件扩展名列表，默认为None表示包含所有文件
    """
    import zipfile
    import os
    from pathlib import Path
    
    data_dir = Path(data_dir)
    
    # 如果没有指定文件扩展名，则不受限制包含所有文件
    if file_exts is None:
        file_exts = []
    
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(data_dir):
            for file in files:
                # 如果file_exts为空，则包含所有文件；否则检查文件扩展名是否在指定列表中
                if not file_exts or any(file.lower().endswith(ext) for ext in file_exts):
                    file_path = os.path.join(root, file)
                    # 计算相对路径，保持目录结构
                    arc_name = os.path.relpath(file_path, data_dir)
                    zipf.write(file_path, arc_name)



# 解压指定数据
def unzip_data(zip_file_path, output_dir):
    """
    解压zip文件到指定目录
    
    参数:
        zip_file_path: zip文件路径
        output_dir: 解压输出目录
    """
    import zipfile
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(output_dir)