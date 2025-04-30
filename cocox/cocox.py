from __future__ import annotations
from dataclasses import dataclass, fields
from collections import defaultdict
from pathlib import Path
import random
import re
import tempfile
from typing import Callable, Dict, List, Optional, Union, Any,Tuple
import numpy as np
from pydantic import BaseModel, Field
import os, json, shutil, copy, cv2
import matplotlib.pyplot as plt


from cocox.base import COCO

from cocox.utils import logger, CCX, STATIC_DATA, plot_anno_info, plot_summary


class COCOX(BaseModel):
    cfg: CCX = Field(default_factory=CCX)
    data: Optional[CCX] = Field(default=None)
    other_data: Optional[dict] = Field(default=None)
    # static_data: Dict = Field(
    #     default_factory=lambda: {
    #         "imgs": 0,             # 图片总数
    #         "anns": 0,             # 标注总数
    #         "cats": {},            # 每个类别的标注数量统计 {类别名: 标注数量}
    #         "img_anns": {},        # 每张图片的标注数量统计 {图片名: 标注数量}
    #         "empty_imgs_num": 0,   # 没有标注的图片数量
    #         "empty_imgs": [],      # 没有标注的图片ID列表
    #         "missing_imgs": [],    # 标注文件中存在但实际目录中不存在的图片列表
    #         "missing_imgs_num": 0, # 缺失图片的数量
    #         "extra_imgs": [],      # 目录中存在但标注文件中不存在的图片列表
    #         "extra_imgs_num": 0    # 多余图片的数量
    #     }
    # )
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, 
                 data:Optional[Union[dict, COCO, COCOX, Path, str]] = None,
                 cfg: Optional[CCX] = None,
                 **kwargs):
        """
        COCOX类的初始化函数，用于处理各种类型的数据源和配置信息
        
        参数:
            data: 数据源，可以是字典、COCO对象、COCOX对象、路径或字符串
            cfg: 配置信息，CCX类型
            **kwargs: 其他参数
                correct_data: 是否校正数据，默认为False
                save_static: 是否保存统计信息，默认为False
                static_path: 统计信息保存路径
        
        功能:
            1. 根据不同类型的数据源初始化数据
                1.1 如果data为字典，coco的json文件内容
                1.2 如果data为COCO对象，从COCO对象中获取
                1.3 如果data为COCOX对象，用于拷贝对象，需要注意原始图片路径问题
                1.4 如果data为路径或字符串，coco的json文件路径
            2. 使用传入的配置更新默认配置
            3. 统计数据集信息
            4. 校正数据（如果需要）
        """
        super().__init__(**kwargs)
        # 参数初始化
        correct_data = kwargs.get("correct_data",False)
        save_static = kwargs.get("save_static",False)
        static_path = Path(kwargs.get("static_path")) if kwargs.get("static_path") is not None else None
        
        # 初始化数据
        res = self._any2data(data)
        if res is None:
            return

        # 使用传入的cfg更新self.cfg
        if cfg:
            self._update_cfg(cfg)
        # if self.cfg.IMGDIR_SRC is None:
        #     self.cfg.IMGDIR_SRC = self.cfg.ROOT.joinpath(self.cfg.IMGDIR) 
        if self.cfg.ROOT == Path("."):
            logger.warning("ROOT is default, please check the data source!!!")
        

        # Statistics for annotation file data
        if self.data is not None:
            if isinstance(data,(Path,str)):

                static_data = self.static(save_static,static_path)
                if not correct_data:
                    logger.info(f"{self.cfg.ROOT}[{type(data).__name__}]: "
                            f"Total Images:{static_data['imgs']}, "
                            f"Total Annotations:{static_data['anns']}, "
                            f"Total Categories:{len(static_data['cats'])}, "
                            f"Annotated Images:{static_data['img_in_ann']}, "
                            f"Images in Folder:{static_data.get('img_in_folder',0)}")
        
        # Correct data
        if correct_data:
            self._correct_validate_data()
            static_data = self.static(save_static,static_path)
            if isinstance(save_static,(Path,str)):
                logger.info(f"{self.cfg.ROOT}[{type(data).__name__}]: "
                            f"Total Images:{static_data['imgs']}, "
                            f"Total Annotations:{static_data['anns']}, "
                            f"Total Categories:{len(static_data['cats'])}, "
                            f"Annotated Images:{static_data['img_in_ann']}, "
                            f"Images in Folder:{static_data.get('img_in_folder',0)}")
            
    def _update_cfg(self, cfg: CCX) -> None:
        """使用传入的cfg更新self.cfg，只更新非默认值的字段
        
        参数:
            cfg: 包含新配置值的CCX对象
            
        返回:
            None
            
        功能:
            1. 遍历传入cfg对象的所有字段
            2. 对于非None的字段值，更新到当前对象的cfg中
            3. 保留注释掉的代码以便未来可能的修改（仅更新非默认值的逻辑）
        """
        default_cfg = CCX()  # 创建默认配置对象
        
        for field in fields(cfg):
            field_name = field.name
            field_value = getattr(cfg, field_name, None)  # 添加默认值None
            default_value = getattr(default_cfg, field_name)
            
            # 只有当字段值不为None且不等于默认值时才更新
            # if field_value is not None and field_value != default_value:
                # setattr(self.cfg, field_name, Path(field_value))
            
            # 直接使用初始化的结果进行更新
            if field_value is not None:
                setattr(self.cfg, field_name, field_value)
        
    def _any2data(self, data: Optional[Union[dict, COCO, COCOX, Path, str]], **kwargs) -> Union[bool, None]:
        """
        将各种类型的输入数据转换为COCO类对象
        
        参数:
            data: 输入数据，可以是字典、COCO对象、COCOX对象、Path对象或字符串路径
            **kwargs: 额外参数
                use_yolo: 是否使用YOLO格式，默认为False
                
        返回:
            成功返回True，失败返回None
            
        功能:
            1. 根据不同类型的输入数据进行相应的处理:
               - COCOX对象: 深拷贝数据和配置
               - 字典: 转换为COCO对象
               - COCO对象: 直接使用
               - 路径或字符串: 根据路径类型进行处理
               - None: 创建空对象
            2. 对于路径类型，会根据路径是否存在、是目录还是文件进行不同处理
            3. 对于JSON文件，会初始化配置信息
        """
        if isinstance(data, COCOX):
            # 从COCOX对象复制数据和配置
            self.data = copy.deepcopy(data.data)
            self.cfg = copy.deepcopy(data.cfg)
            self.cfg.IMGFOLDER = copy.deepcopy(data.cfg.IMGFOLDER)
            self.cfg.IMGDIR_SRC = copy.deepcopy(data.cfg.IMGDIR_SRC)
        elif isinstance(data, dict):
            # 将字典转换为COCO对象
            self.data = self.dict2_(data)
            # TODO:需不需要对配置进行初始化？
        elif isinstance(data, COCO):
            # 直接使用COCO对象
            self.data = data
        elif isinstance(data, (Path, str)):
            # 处理路径类型的输入
            data = Path(data)
            if data.exists():
                # 路径存在的情况
                if data.is_dir():
                    # TODO:目录处理，暂不支持YOLO模式
                    return None
                elif data.is_file() and data.suffix == '.json':
                    # JSON文件处理
                    self.data = COCO(data)
                    # 初始化配置
                    IMGFOLDER = Path(data.stem[data.stem.find('_')+1:] if data.stem.startswith('instances_') else data.stem)
                    
                    self.cfg = CCX(
                        ROOT=data.parent.parent,
                        ANNDIR=Path(data.parent.name),
                        ANNFILE=Path(data.name),
                        IMGFOLDER=IMGFOLDER
                    )
                    # 根据配置更新原始目录,同时会修改self.cfg.IMGFOLDER
                    self.cfg.IMGDIR_SRC = self.set_imgdir()
                    
                    if not self.cfg.ANNFILE.stem.startswith('instances'):
                        logger.info(f"Suggestion: please check the annotation file name, it should start with 'instances_'!")
                else:
                    logger.warning(f"Path {data} is not a json file, please check the data source!!!")
                    return None
            else:
                # 不存在，则路径为ROOT
                logger.info(f"Path {data} not found, create an empty COCOX object!")
                os.makedirs(data,exist_ok=True)
                # 对cfg进行初始化
                self.data = None
                # 根据路径类型设置配置
                if data.is_dir():
                    self.cfg = CCX(ROOT=data)
                else:
                    self.cfg = CCX(ROOT=data.parent.parent)
        elif data is None:
            # 无数据输入，创建空对象
            self.data = None
            logger.info("No data provided, creating an empty COCOX object.")
        else:
            raise ValueError(f"Invalid data: {data}")
        return True
 
    def dict2_(self, data: dict) -> COCO:
        """
        将字典数据转换为COCO对象
        
        Args:
            data (dict): COCO格式的字典数据，包含images、annotations和categories
            
        Returns:
            COCO: 转换后的COCO对象
            
        Notes:
            该函数通过创建临时JSON文件来实现字典到COCO对象的转换
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            # Write the dictionary to a temporary JSON file
            json.dump(data, temp_file)
            temp_file_name = temp_file.name

        # Create CC object using the temporary JSON file
        return COCO(temp_file_name)
       
    def set_imgdir(self)->Path:
        """
        获取可用的图片目录,不会无中生有
        
        返回:
            Path: 返回有效的图片目录路径，如果不存在则返回None
            
        功能:
            该函数会检查图片目录是否存在，并根据实际情况调整IMGFOLDER的值
            - 如果ROOT/IMGDIR/IMGFOLDER存在，则返回该路径
            - 如果ROOT/IMGDIR存在，则将IMGFOLDER设置为"."并返回ROOT/IMGDIR
            - 如果都不存在，返回None
        """
        temp_imgdir = self.cfg.ROOT.joinpath(self.cfg.IMGDIR)
        temp_imgdir_with_folder = temp_imgdir.joinpath(self.cfg.IMGFOLDER)
        
        if temp_imgdir_with_folder.exists():
            return temp_imgdir_with_folder
        elif temp_imgdir.exists():
            self.cfg.IMGFOLDER =Path(".")
            return temp_imgdir
        else:
            return None
        
    def _correct_validate_data(self)->None:             
        # Only keep data without differences
        # if self.static_data["missing_imgs_num"] > 0:
        #     # Filter out annotations for missing images
        #     valid_imgs = [img for img in self.data.dataset['images'] 
        #                  if img['file_name'] not in self.static_data["missing_imgs"]]
        #     valid_img_ids = set(img['id'] for img in valid_imgs)
            
        #     valid_anns = [ann for ann in self.data.dataset['annotations'] 
        #                  if ann['image_id'] in valid_img_ids]
            
        #     self.data.dataset['images'] = valid_imgs
        #     self.data.dataset['annotations'] = valid_anns
        #     self.data.createIndex()
        pass    

    def _2dict(self)->dict:
        """将数据转换为字典"""
        if self.other_data is None:
            self.other_data = {}
        
        if not self.other_data.get('dict'):
            self.other_data['dict'] = {
                'data': {},  # 初始化data字典
                'id2cls': {}  # 如果需要，也可以初始化其他必要的键
            }
        
        # Ensure self.data exists
        assert self.data is not None, "Dataset is empty, please load data first"
        self.other_data['dict']['data'] = {
            'images': list(self.data.imgs.values()),
            'annotations': list(self.data.anns.values()),
            'categories': list(self.data.cats.values())
        }
        self.other_data['dict']['id2cls'] = {cat['id']: cat['name'] for cat in self.data.dataset['categories']}
        
    # def _2yolo(self)->None:
    #     """将数据转换为YOLO格式"""
    #     if self.other_data is None:
    #         self.other_data = {}
        
    #     if not self.other_data.get('yolo'):
    #         self.other_data['yolo'] = {
    #             'data': {},  # 初始化data字典
    #             'id2cls': {}  # 如果需要，也可以初始化其他必要的键
    #         }
        
    #     # Ensure self.data exists
    #     assert self.data is not None, "Dataset is empty, please load data first"
    #     # Create image dict
    #     images = {"%g" % x["id"]: x for x in self.data.dataset["images"]}
    #     # Create image-annotations dict
    #     imgToAnns = defaultdict(list)
    #     for ann in self.data.dataset.get("annotations",[]):
    #         imgToAnns[ann["image_id"]].append(ann)

    #     # Write labels file
    #     for img_id, img in images.items():
    #         f = img["file_name"]
            
    #         # 如果该图片有标注数据
    #         if int(img_id) in imgToAnns:
    #             h, w = img["height"], img["width"]
    #             bboxes = []
    #             for ann in imgToAnns[int(img_id)]:
    #                 # The COCO box format is [top left x, top left y, width, height]
    #                 box = np.array(ann["bbox"], dtype=np.float64)
    #                 box[:2] += box[2:] / 2  # xy top-left corner to center
    #                 box[[0, 2]] /= w  # normalize x
    #                 box[[1, 3]] /= h  # normalize y
    #                 if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
    #                     continue

    #                 cls = ann["category_id"] - 1 
    #                 box = [cls] + box.tolist()
    #                 if box not in bboxes:
    #                     bboxes.append(box)
    #             self.other_data['yolo']['data'][f] = bboxes
    #         # else:
    #         #     # 如果图片没有标注数据，仍然保存空列表
    #         #     self.other_data['yolo']['data'][f] = []
                
    #     self.other_data['yolo']['id2cls'] = {cat['id']-1: cat['name'] for cat in self.data.dataset['categories']}

    # def _get_cat(self,cat:Union[int,str],return_id:bool=False):
    #     """获取类别信息, 用来判断类别是否存在
    #     force_int: 强制返回int类型
    #     """
    #     if isinstance(cat,int):
    #         if return_id:
    #             return True, cat
    #         cat_str = self.data.cats.get(cat,None)
    #         if cat_str:
    #             return True, cat_str.get('name',None)
    #         else:
    #             return False, None
    #     elif isinstance(cat,str):
    #         for c in self.data.cats.values():
    #             if c['name'] == cat:
    #                 return True, c.get('id',None)
    #         return False, None
    #     else:
    #         raise ValueError(f"Invalid category: {cat}")
        
    def _get_img(self,img:Union[int,str],return_id:bool=False):
        """
        获取图片信息, 用来判断图片是否存在
        支持图片路径的模糊搜索
        """
        if isinstance(img,int):
            if return_id:
                return True, img
            img_str = self.data.imgs.get(img,None)
            if img_str:
                return True, img_str.get('file_name',None)
            else:
                return False, None
        elif isinstance(img,str):
            for img_id in self.data.imgs.values() :
                if img_id['file_name'][:len(img)] == img:
                    return True, img_id.get('id',None)
            return False, None
        else:
            raise ValueError(f"Invalid image: {img}")
    
    # def _get_imglist(self)->List[str]:
    #     """获取图片列表"""
    #     return [img['file_name'] for img in self.data.dataset['images']] if self.data else []
    
    # def _get_catlist(self)->List[str]:
    #     """获取类别列表"""
    #     return [cat['name'] for cat in self.data.dataset['categories']]


    # def del_empty_dir(self)->None:
    #     """删除空目录"""
    #     for dir in self.cfg.ROOT.iterdir():
    #         if dir.is_dir() and not any(dir.iterdir()):
    #             try:
    #                 dir.rmdir()
    #             except Exception as e:
    #                 logger.error(f"删除目录 {dir} 时出错: {e}")
    
    def vis_anno_info(self, save_dir=Path("")):
        """从COCO数据集绘制标签统计信息"""
        save_dir = self.cfg.ROOT.joinpath(self.cfg.IMGFOLDER) if save_dir is None else save_dir
        
        # 获取所有标注信息
        data = copy.deepcopy(self.data)
        anns = data.loadAnns(data.getAnnIds())
        boxes_raw = np.array([x['bbox'] for x in anns])
        img_ids = data.getImgIds()
        img_info = data.loadImgs(img_ids)
        
        # 创建归一化后的边界框数组
        boxes = np.zeros_like(boxes_raw, dtype=np.float32)
        for i, ann in enumerate(anns):
            img = next((img for img in img_info if img['id'] == ann['image_id']), None)
            if img:
                # 将边界框坐标归一化到0-1范围
                boxes[i, 0] = boxes_raw[i, 0] / img['width']  # x
                boxes[i, 1] = boxes_raw[i, 1] / img['height']  # y
                boxes[i, 2] = boxes_raw[i, 2] / img['width']   # width
                boxes[i, 3] = boxes_raw[i, 3] / img['height']  # height
        
        cls = np.array([x['category_id']-1 for x in anns])
        names = {cat['id']: cat['name'] for cat in data.loadCats(data.getCatIds())}

        static_data = self._static()   
        plot_summary(static_data, save_dir)
        json_static_data = plot_anno_info(boxes, cls, names, save_dir)
        logger.info(f"annotation visualization results saved to {save_dir}")
        return json_static_data


    def _static(self)->STATIC_DATA:
        """
        统计数据集信息，返回STATIC_DATA对象
        
        参数:
            save: 是否保存统计结果到文件
            static_path: 保存路径，如果为None则保存到数据集根目录
            
        返回:
            STATIC_DATA: 包含数据集统计信息的对象
        """
        static_data = STATIC_DATA()
        

        # 图片文件夹
        img_dir = self.cfg.ROOT.joinpath(self.cfg.IMGDIR).joinpath(self.cfg.IMGFOLDER)
        if img_dir.exists():
            list_image_in_dir = [f.name for f in img_dir.iterdir() if f.is_file() and f.suffix in ['.jpg','.png','.jpeg','.bmp','.tiff','.gif']]
        else:
            list_image_in_dir = []
        static_data.nums_image_in_dir = len(list_image_in_dir)
        
        # 图片
        dict_image_in_img = {img['id']: img['file_name'] for img in self.data.dataset['images']}
        list_image_in_img = list(dict_image_in_img.values())
        static_data.nums_image_in_img = len(dict_image_in_img)
        # 标注的图片，多个标注对应一张图片
        list_image_in_ann = set([dict_image_in_img[ann['image_id']] for ann in self.data.dataset['annotations']])
        
    
        
        static_data.nums_image_in_ann = len(list_image_in_ann)
        # 类别 id: 类别名
        dict_cats = {cat['id'] : cat['name'] for cat in self.data.dataset['categories']}
        static_data.cats = dict_cats
        static_data.nums_cats = len(list(dict_cats.keys()))
    
        # 类别有多少图片,多少个box
        result = {}
        for ann in self.data.dataset['annotations']:
            img_name = dict_image_in_img[ann['image_id']]
            cat_name = dict_cats[ann['category_id']]
            if dict_cats[ann['category_id']] not in result.keys():
                result[cat_name] = {'img':[],'box':0}
            result[cat_name]['img'].append(img_name)
            result[cat_name]['box'] += 1
        # 对每个类别的图片列表进行去重，确保每张图片只被计算一次
        for cat_name in result:
            result[cat_name]['img'] = list(set(result[cat_name]['img']))
        static_data.relate_cat_box = {k:v['box'] for k,v in result.items()}
        static_data.relate_cat_img = {k:v['img'] for k,v in result.items()}
    
        
        # 图片与标注的差异
        list_img_ann_diff_in_img = set(list_image_in_img) - set(list_image_in_ann)
        list_img_ann_diff_in_ann = set(list_image_in_ann) - set(list_image_in_img)
        static_data.nums_img_ann_diff = len(list_img_ann_diff_in_img) + len(list_img_ann_diff_in_ann)
        static_data.list_img_ann_diff_in_img = list(list_img_ann_diff_in_img)
        static_data.list_img_ann_diff_in_ann = list(list_img_ann_diff_in_ann)
        
        # 图片与文件夹的差异
        if len(list_image_in_dir) > 0:
            list_img_dir_diff_in_img = set(list_image_in_dir) - set(list_image_in_img)
            list_img_dir_diff_in_ann = set(list_image_in_img) - set(list_image_in_dir)
            static_data.nums_img_dir_diff = len(list_img_dir_diff_in_img) + len(list_img_dir_diff_in_ann)
            static_data.list_img_dir_diff_in_img = list(list_img_dir_diff_in_img)
            static_data.list_img_dir_diff_in_ann = list(list_img_dir_diff_in_ann)
        
        # 图标标注
        nums_anns = len(self.data.dataset['annotations'])
        static_data.nums_anns = nums_anns
        
        return static_data
        

    def static(self, save:bool=False, static_path:Optional[Path]=None)->Optional[Dict]:
        # 统计数据，包括：
        # 1. 图片总数 imgs
        # 2. 标注总数 anns
        # 3. 类别统计 cats
        # 4. 标注的图片总数 anno_imgs
        # *5. 文件夹中图片总数 folder_imgs
        
        #########################################
        # 7. 实际标注文件与标注图片数量差异以及图片列表 folder_ann_diff
        # 8. 实际标注文件与文件夹中图片数量差异以及图片列表 folder_img_diff
        
        """统计数据"""
        static_data = {
            'cats': {},
            'imgs_list': [],
            'imgs': 0,
            'anns': 0,
            'img_in_ann_list': [],
            'img_in_ann': 0,
            'img_in_folder_list': [],
            'img_in_folder': 0
        }
        if not self.data:
            logger.warning("数据为空,跳过统计")
            return static_data
            
        if not self.data.dataset.get('images'):
            logger.warning("图片数据为空,跳过图片统计")
        else:
            # 统计图片和标注总数
            imgs = [img['file_name'] for img in self.data.dataset['images']]
            static_data["imgs_list"] = imgs
            static_data["imgs"] = len(imgs)
            
        if not self.data.dataset.get('annotations'):
            logger.warning("标注数据为空")
        
        static_data["anns"] = len(self.data.dataset['annotations'])
        
        # 统计每个类别的标注数量
        cat_ann_counts = {}
        for ann in self.data.dataset['annotations']:
            cat_id = ann['category_id']
            cat_ann_counts[cat_id] = cat_ann_counts.get(cat_id, 0) + 1
            
        if not self.data.dataset.get('categories'):
            logger.warning("类别数据为空,跳过类别统计")
        else:
            for cat in self.data.dataset['categories']:
                static_data["cats"][cat['name']] = cat_ann_counts.get(cat['id'], 0)

        # 统计每张图片的标注数量，并计算单个图片的标注的最大最小数量以及整理的标注的平均值
        static_data["img_ann_counts"] = {}
        ann_counts = []
        for ann in self.data.dataset['annotations']:
            img_name = self._get_img(ann['image_id'])[1]
            if img_name:
                static_data["img_ann_counts"][img_name] = static_data["img_ann_counts"].get(img_name, 0) + 1
                ann_counts.append(static_data["img_ann_counts"][img_name])
            else:
                logger.warning(f"找不到图片ID {ann['image_id']} 对应的图片名称")
                
        static_data["ann_in_one_img"] = {
            'min': min(ann_counts) if ann_counts else 0,
            'max': max(ann_counts) if ann_counts else 0,
            'mean': sum(ann_counts) / len(ann_counts) if ann_counts else 0
        }

        # 标注的图片总数 anno_imgs
        anno_imgs = list(static_data["img_ann_counts"].keys())  # 使用已创建的字典
        static_data["img_in_ann_list"] = anno_imgs
        static_data["img_in_ann"] = len(anno_imgs)
        
        # 文件夹中图片总数 folder_imgs
        img_dir = self.cfg.ROOT.joinpath(self.cfg.IMGDIR)
        if not img_dir.exists():
            logger.warning(f"图片目录 {img_dir} 不存在,跳过文件夹统计")
        else:
            folder_imgs = set([f.name for f in img_dir.iterdir() if f.is_file()])
            static_data["img_in_folder_list"] = list(folder_imgs)
            static_data["img_in_folder"] = len(folder_imgs)
                    
        # 按照数值类型排序输出
        # 先输出单一数值的键值对
        sorted_data = {}
        list_data = {}
        
        # 分别存储不同类型的数据
        int_float_data = {}
        dict_data = {}
        list_data = {}
        
        # 按类型分类数据
        for key, value in static_data.items():
            if isinstance(value, (int, float)):
                int_float_data[key] = value
            elif isinstance(value, dict):
                dict_data[key] = value
            elif isinstance(value, list):
                list_data[key] = value
                
        # 按指定顺序合并数据
        static_data = {
            **dict(sorted(int_float_data.items(), key=lambda x: x[1], reverse=True)),
            **dict(sorted(dict_data.items())),
            **dict(sorted(list_data.items()))
        }
        
        # Check if number of annotated images matches total images
        error_info = ''
        if static_data["img_in_ann"] != static_data["imgs"]:
            diff = abs(static_data["img_in_ann"] - static_data["imgs"])
            error_info = f"{self.cfg.ROOT}/{self.cfg.ANNFILE}: Annotated images ({static_data['img_in_ann']}) != Total images ({static_data['imgs']}), Diff nums: {diff}"
            logger.warning(error_info)
        
        if save:
            if static_path is None:
                file_path = self.cfg.ROOT.joinpath(f"static_{str(self.cfg.ANNFILE)}")
                
            else:
                os.makedirs(static_path, exist_ok=True)
                safe_name = str(self.cfg.ROOT).replace('/', '-').replace('\\', '-')
                file_path = static_path.joinpath(f"static_{safe_name}_{str(self.cfg.ANNFILE)}")
                txt_path = static_path.joinpath(f"error_list.txt")
                if error_info:
                    with open(txt_path, 'a+', encoding='utf-8') as txt_file:
                        txt_file.write(error_info+"\n")
                    
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(static_data, json_file, ensure_ascii=False, indent=2)
        return static_data
    
    def name_or_dir(self,name:Optional[Union[Path,str]]=None)->Union[str]:
        # 判断是否为完整路径
        if "/" in str(name) or "\\" in str(name):
            # 包含路径分隔符,是完整路径
            return 'dir'
        else:
            # 不包含路径分隔符,只是文件夹名称
            return 'name'
    
    # def save_yolo(self,dst_dir:Optional[Union[Path,str]]=None,overwrite:bool=True)->None:
    #     """
    #     保存YOLO数据
    #     dst_dir: 保存路径，默认在ROOT/yolo
    #     dst_name: 保存名称，默认在yolo
    #     """
    #     if self.other_data is None or not self.other_data.get('yolo'):
    #         self._2yolo()
        
    #     # 根据输入路径或默认配置确定YOLO输出路径
    #     target_path = dst_dir if dst_dir else self.cfg.YOLODIR
    #     yolo_path = target_path if self.name_or_dir(target_path) == 'dir' else self.cfg.ROOT.joinpath(target_path)

    #     if overwrite and yolo_path.exists():
    #         shutil.rmtree(yolo_path)
    #     yolo_path.mkdir(parents=True,exist_ok=True)
        
    #     assert len(self.other_data['yolo']['data']), logger.error(f"YOLO data is empty in {yolo_path}, please convert to YOLO format first by calling self._2yolo()")
    #     for filename,labels in self.other_data['yolo']['data'].items():
    #         src_path = self.cfg.IMGDIR_SRC.joinpath(filename)
    #         dst_path = yolo_path.joinpath(filename)
    #         if src_path.exists():
    #             shutil.copy2(src_path,dst_path)
    #         else:
    #             logger.warning(f"Image not found: {src_path}")
            
    #         label_path = yolo_path.joinpath(Path(filename).with_suffix(".txt"))
    #         # Write
    #         if labels:
    #             with open(label_path, "w") as file:
    #                 for label in labels:
    #                     line = (*label,)  # cls, box or segments
    #                     file.write(("%g " * len(line)).rstrip() % line + "\n")
    #         else:
    #             # logger.warning(f"Label is empty: {label_path}")
    #             pass

    def _get_img_id(self,img_path:Union[str,Path])->int:
        """获取图片ID"""
        img_path = Path(img_path)
        # 全局变量
        # 使用other_data字典存储map_img_id
        if self.other_data is None:
            self.other_data = {}
        if 'map_img_id' not in self.other_data:
            self.other_data['map_img_id'] = {it['file_name'] : it['id'] for it in self.data.dataset['images']}
            
        return  self.other_data['map_img_id'][img_path.name]

    # 单个图像显示
    def _vis_gt(self,img_path:Union[str,Path],dst_dir:Path,overwrite:bool=True)->None:
        """可视化数据"""
        target_path = dst_dir if dst_dir else self.cfg.IMGDIR_VISUAL
        visual_dir = target_path if self.name_or_dir(target_path) == 'dir' else self.cfg.ROOT.joinpath(target_path)
            
        if overwrite and visual_dir.exists():
            shutil.rmtree(visual_dir)
        visual_dir.mkdir(parents=True,exist_ok=True)
        
       
        img_path = Path(img_path)
        img_path = self.cfg.ROOT.joinpath(self.cfg.IMGDIR).joinpath(img_path)
        out_path = visual_dir.joinpath(img_path.name)
        
        image = cv2.imread(img_path)
        plt.imshow(image) 
        plt.axis('off')
        # 图片获取图片id
        img_id = self._get_img_id(img_path.name)
        # 根据图片获取所有anno
        ann_ids = self.data.getAnnIds(imgIds=img_id)
        one_anns = [self.data.anns[i] for i in ann_ids]
        self.data.showBBox(anns=one_anns)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)


    def vis_gt(self,dst_dir:Optional[Union[Path,str]]=None,overwrite:bool=True)->None:
        """可视化数据"""

        # 根据输入路径或默认配置确定YOLO输出路径
        target_path = dst_dir if dst_dir else self.cfg.IMGDIR_VISUAL
        visual_dir = target_path if self.name_or_dir(target_path) == 'dir' else self.cfg.ROOT.joinpath(target_path)
            
        if overwrite and visual_dir.exists():
            shutil.rmtree(visual_dir)
        visual_dir.mkdir(parents=True,exist_ok=True)
        
        assert self.data is not None, logger.error("Data is None, please load data first")
        
        for img_id in self.data.getImgIds() :
            file_name = self.data.imgs[img_id]['file_name']
            img_path = self.cfg.ROOT.joinpath(self.cfg.IMGDIR).joinpath(file_name)
            out_path = visual_dir.joinpath(file_name)
            
            # 获取该图片所有的anno
            anno_ids = self.data.getAnnIds(imgIds=img_id)
            one_anns = [self.data.anns[i] for i in anno_ids]
            
            image = cv2.imread(img_path)
            plt.imshow(image) 
            plt.axis('off')
            self.data.showBBox(anns=one_anns)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        logger.info(f"Visual image completed. Output saved to {visual_dir}")

    # def save_annfile(self,annfile:Optional[Path]=None)->None:
        
    #     """保存annfile"""
    #     if self.other_data is None or not self.other_data.get('dict'):
    #         self._2dict()
    #     ann_path = self.cfg.ROOT.joinpath(self.cfg.ANNDIR)
    #     ann_path.mkdir(parents=True,exist_ok=True)
    #     with open(ann_path.joinpath(annfile if annfile else self.cfg.ANNFILE), 'w', encoding='utf-8') as json_file:
    #         json.dump(self.other_data['dict']['data'], json_file, ensure_ascii=False, indent=2)  # Use indent parameter to beautify output

    # def save_img(self,dst_path:Optional[Path]=None):
    #     """复制图片到目标路径"""
    #     try:
    #         if not self.cfg.IMGDIR_SRC.exists():
    #             logger.error(f"IMGDIR_SRC not found: {self.cfg.IMGDIR_SRC}")
    #             return
    #     except Exception as e:
    #         logger.error(f"检查IMGDIR_SRC时出错: {str(e)}")
    #         return
        
    #     # Get image list and create destination path
    #     imglist = self._get_imglist()
    #     img_dst_path = self.cfg.ROOT.joinpath(self.cfg.IMGDIR) if dst_path is None else dst_path
    #     img_dst_path.mkdir(parents=True, exist_ok=True)
        
    #     # Batch copy images
    #     for img in imglist:
    #         src_path = self.cfg.IMGDIR_SRC.joinpath(img)
    #         temp_path = img_dst_path.joinpath(img)
            
    #         # Skip if destination file exists
    #         if temp_path.exists():
    #             continue
                
    #         # Copy image file
    #         if src_path.exists():
    #             # Ensure destination directory exists
    #             temp_path.parent.mkdir(parents=True, exist_ok=True)
    #             shutil.copy2(src_path, temp_path)
    #         else:
    #             logger.warning(f"Image {img} not found in source directory {self.cfg.IMGDIR_SRC}")
                
    #     # Update image source path to destination path
    #     self.cfg.IMGDIR_SRC = img_dst_path

    # def save_data(self,
    #               dst_file: Optional[CCX] = None,
    #               visual: bool = False, 
    #               yolo: bool = False,
    #               only_ann: bool = False,
    #               overwrite: bool = True) -> Optional[COCOX]:
    #     """将数据保存为实际文件

    #     Args:
    #         dst_file (Optional[CCX], optional): 目标配置文件,用于指定保存路径. Defaults to None.
    #         visual (bool, optional): 是否生成可视化图片. Defaults to False.
    #         yolo (bool, optional): 是否保存为YOLO格式. Defaults to False.
    #         only_ann (bool, optional): 是否只保存标注文件. Defaults to False.
    #         overwrite (bool, optional): 是否覆盖已有文件. Defaults to True.

    #     Returns:
    #         Optional[COCOX]: 返回保存后的COCOX对象,如果dst_file为None则返回self
    #     """
    #     assert self.data, logger.error("Data is None")
        
    #     # 如果指定了目标配置文件
    #     # 选择操作对象(新建或当前)
    #     obj = COCOX(data=self, cfg=dst_file) if dst_file else self
    #     # obj.cfg.IMGDIR_SRC = self.cfg.ROOT.joinpath(self.cfg.IMGDIR)
    #     # 执行保存操作
    #     if not only_ann:
    #         obj.save_img()
    #     obj.save_annfile()
    #     if yolo:
    #         obj.save_yolo()
    #     if visual:
    #         obj.visual(overwrite=overwrite)
        
    #     # 清空多余目录
    #     obj.del_empty_dir()    
    #     return obj
        



    # def update_cat(self, new_cat:Dict[int,str]):
    #     """
    #     根据给定的类别字典更新数据集中的类别
    #     """
    #     if self.data is None:
    #         return
    #     dataset = self.data.dataset
    #     # Check if categories need to be expanded
    #     raw_cat = {cat['id']: cat['name'] for cat in dataset['categories']}
    #     rawId = max(raw_cat.keys())
        
    #     existing_names = {cat['name'] for cat in dataset['categories']}
    #     for _, new_name in new_cat.items():
    #         if new_name not in existing_names:
    #             rawId += 1
    #             dataset['categories'].append({
    #                 'id': rawId,
    #                 'name': new_name,
    #                 'supercategory': ''
    #             })
    #             logger.info(f"Added new category: ID {rawId}, name '{new_name}'")
       
    #     new_cat = {cat['id']: new_id for cat in dataset['categories'] for new_id, name in new_cat.items() if cat['name'] == name}
        
    #     # Store unique mapping info
    #     mapping_info = set()
            
    #     # Update categories
    #     for category in dataset['categories']:
    #         if category['id'] in new_cat.keys():
    #             old_id = category['id']
    #             old_name = category['name']
    #             category['id'] = new_cat[old_id]
    #             if old_id != category['id'] or old_name != category['name']:
    #                 mapping_info.add(f"[{old_name}] {old_id} -> {category['id']}")

    #     # Update annotations
    #     for annotation in dataset['annotations']:
    #         if annotation['category_id'] in new_cat.keys():
    #             old_id = annotation['category_id']
    #             annotation['category_id'] = new_cat[old_id]
                
    #     # Log unique mappings at once
    #     if mapping_info:
    #         logger.info("Category mapping:\n" + " | ".join(sorted(mapping_info)))
            
    #     self.data.createIndex()

    # def update_cat_force(self,new_cat:Dict[int,str]):
    #     """
    #     强制更新类别
    #     """
    #     dataset = self.data.dataset
    #     # 清空原有类别
    #     dataset['categories'] = []
    #     # 添加新类别
    #     for cat_id, cat_name in new_cat.items():
    #         dataset['categories'].append({
    #             'id': cat_id,
    #             'name': cat_name,
    #             'supercategory': ''
    #         })
    #     # 更新索引
    #     self.data.createIndex()

    # def rename_cat(self, raw_cat:str, new_cat:str)->None:
    #     """
    #     修改类别名称,不对ID进行修改
    #     """
    #     dataset = self.data.dataset
    #     for category in dataset['categories']:
    #         if category['name'] == raw_cat:
    #             category['name'] = new_cat
    #     self.data.createIndex()
    
    # def align_cat(self,other_cat:Dict,cat_keep:bool=True)->Dict:
    #     """
    #     对齐两个数据集的类别，对id做处理，但未对ann数据做处理
        
    #     Args:
    #         other_cat: 另一个数据集的类别字典
    #         cat_keep: 是否保留当前数据集的类别作为基准
    #                  True: 以当前数据集类别为基准,将other_cat中新的类别添加进来
    #                  False: 以other_cat为基准,将当前数据集中缺失的类别添加到other_cat中
    #     """
    #     raw_cat = {cat['id']: cat['name'] for cat in self.data.dataset['categories']} if self.data else {}
    #     if cat_keep:
    #         new_cat = raw_cat
    #         maxId = len(raw_cat.keys())
    #         new_cats = [cat for cat in other_cat if cat not in raw_cat]
    #         logger.info(f"Found {len(new_cats)} new categories from other dataset")
    #         for cat in other_cat:
    #             if cat not in raw_cat:
    #                 maxId += 1
    #                 raw_cat[maxId] = other_cat[cat]
    #     else:
    #         new_cat = other_cat
    #         maxId = len(raw_cat.keys())
    #         missing_cats = [cat for cat in raw_cat if cat not in other_cat]
    #         logger.info(f"Found {len(missing_cats)} categories missing from other dataset")
    #         for cat in missing_cats:
    #             maxId += 1
    #             new_cat[maxId] = raw_cat[cat]
        
    #     return new_cat        

    # def rename_cat_in_ann(self,old_name:Union[str,int],new_name:Union[str,int]):
    #     """
    #     只修改标注中的类别，类别中的名称不修改
    #     """
    #     _,old_id = self._get_cat(old_name,return_id=True)
    #     _,new_id = self._get_cat(new_name,return_id=True)
    #     assert old_id in [cat['id'] for cat in self.data.dataset['categories']], \
    #         logger.error(f"Category {old_name} not found in dataset")
    #     assert new_id in [cat['id'] for cat in self.data.dataset['categories']], \
    #         logger.error(f"Category {new_name} not found in dataset")
    #     for ann in self.data.dataset['annotations']:
    #         if ann['category_id'] == old_id:
    #             ann['category_id'] = new_id
    #     self.data.createIndex()
    
    # def _updateIndex(self,imgIndex:Optional[int]=None,annIndex:Optional[int]=None):
    #     """
    #     更新数据集索引
    #     """

    #     imgIndex = imgIndex or 1
    #     if imgIndex < 1:
    #         raise ValueError("imgIndex must be greater than 0")
    #     annIndex = annIndex or 1
    #     if annIndex < 1:
    #         raise ValueError("annIndex must be greater than 0")

    #     if not self.data:
    #         return
            
    #     img_id_map = {}
    #     for img in self.data.dataset['images']:
    #         img_id_map[img['id']] = imgIndex
    #         img['id'] = imgIndex
    #         imgIndex += 1
            
    #     for img_ann in self.data.dataset['annotations']:
    #         if img_ann['image_id'] in img_id_map:
    #             img_ann['image_id'] = img_id_map[img_ann['image_id']]
    #         else:
    #             logger.warning(f"Image ID {img_ann['image_id']} not found in img_id_map")


    #     ann_id_map = {} # old: new                
    #     for anns in self.data.dataset['annotations']:
    #         ann_id_map[anns['id']] = annIndex
    #         anns['id'] = annIndex
    #         annIndex += 1
        
    #     self.data.createIndex()


    # def _merge(self,other:COCOX,cat_keep:Optional[bool]=None,overwrite:Optional[bool]=None):
    #     """合并两个数据集,不能为空

    #     Args:
    #         other (COCOX): 要合并的数据集对象
    #         cat_keep (Optional[bool], optional): 类别保留方式. 
    #             True: 以self为主,保留self的类别ID,对other的类别重新编号
    #             False: 以other为主,保留other的类别ID,对self的别重新编号
    #             默认为True
    #         overwrite (Optional[bool], optional): 当发现重复图片时是否覆盖.
    #             True: 使用other中的图片信息覆盖self中的重复图片
    #             False: 保留self中的图片信息
    #             默认为False

    #     Note:
    #         - 该函数会修改self和other的数据
    #         - 合并后的数据存储在self中
    #         - 合并过程包括:
    #             1. 对齐两个数据集的类别
    #             2. 重新编号图片ID和标注ID
    #             3. 合并图片信息和标注信息
    #     """
    #     # 如果self.data为空，直接使用other的数据
    #     if not self.data:
    #         self.data = other.data
    #         return
        
    #     # 如果other.data为空，保持self.data不变
    #     if not other.data:
    #         return
            
    #     cat_keep = cat_keep or True
    #     overwrite = overwrite or False
        
    #     # 获取other数据集的类别字典
    #     other_cat = {cat['id']: cat['name'] for cat in other.data.dataset['categories']}
    #     # 对齐两个数据集的类别
    #     new_cat = self.align_cat(other_cat=other_cat,cat_keep=cat_keep)
        
    #     # 获取当前数据集的图片列表和other数据集的图片ID列表
    #     raw_imglist = self._get_imglist()
    #     other_imgidlist = [img['id'] for img in other.data.dataset['images']]
        
    #     # 更新两个数据集的类别信息
    #     self.update_cat(new_cat=new_cat)
    #     other.update_cat(new_cat=new_cat)
        
    #     # 重新编号图片ID和标注ID
    #     if cat_keep:
    #         # 以self为主,self从1开始编号,other接着编号
    #         self._updateIndex(imgIndex=1,annIndex=1)
    #         other._updateIndex(imgIndex=len(self.data.imgs)+1,annIndex=len(self.data.anns)+1)
    #     else:
    #         # 以other为主,other从1开始编号,self接着编号
    #         other._updateIndex(imgIndex=1,annIndex=1)
    #         self._updateIndex(imgIndex=len(other.data.imgs)+1,annIndex=len(other.data.anns)+1)
        
    #     # 合并图片信息
    #     duplicate_count = 0
    #     for img in other.data.dataset['images']:
    #         if img['file_name'] in raw_imglist:
    #             duplicate_count += 1
    #             if overwrite:
    #                 self.data.dataset['images'].append(img)
    #         else:
    #             self.data.dataset['images'].append(img)
        
    #     if duplicate_count > 0:
    #         logger.debug(f"Found {duplicate_count} duplicate images")

    #     # 合并标注信息
    #     for ann in other.data.dataset['annotations']:
    #         img_id = ann['image_id']
    #         # 获取对应图片的文件名
    #         img_name = next((img['file_name'] for img in other.data.dataset['images'] 
    #                         if img['id'] == img_id), None)
    #         if img_name:
    #             # 检查该图片是否在合并后的数据集中
    #             merged_img = next((img for img in self.data.dataset['images'] 
    #                              if img['file_name'] == img_name), None)
    #             if merged_img:
    #                 # 更新标注的image_id为合并后数据集中的id
    #                 ann['image_id'] = merged_img['id']
    #                 self.data.dataset['annotations'].append(ann)

    #     # 重建索引
    #     self.data.createIndex()
        
    # def merge(self,
    #           others:Union[COCOX,List[COCOX]],
    #           cat_keep:Optional[bool]=None,
    #           overwrite:Optional[bool]=None,
    #           dst_file:Optional[CCX]=None,
    #           save_img:bool=True):
    #     """
    #     Merge multiple datasets into one, self can be empty
    #     Args:
    #         cat_keep: Category retention mode, True: keep self's categories, False: keep other's categories
    #         overwrite: Whether to overwrite when image name exists
    #         dst_file: Destination file configuration
    #         save_img: Whether to save images
    #     Returns:
    #         COCOX: New merged dataset object
    #     """
    #     try:
    #         if save_img and not dst_file:
    #             logger.warning("="*40+"\n"+"Warning: When saving images (save_img=True), dst_file need to be specified to ensure complete data saving"+"\n"+"="*40)
                
    #         if isinstance(others,COCOX):
    #             others = [others]

    #         # Create new object or use current object
    #         obj = COCOX(data=self, cfg=dst_file if dst_file else None)
    #         if obj.data and save_img:
    #             try:
    #                 obj.save_img()
    #             except Exception as e:
    #                 logger.error(f"Failed to save images: {e}")
            
    #         # Initialize statistics data
    #         static_data = {
    #             'imgs': 0,
    #             'anns': 0,
    #             'cats': {},  # Use dict instead of list for category counts
    #             'img_in_ann': 0,
    #             'img_in_folder': 0
    #         }

    #         # Get initial object statistics
    #         if obj.data:
    #             try:
    #                 init_stats = obj.static()
    #                 static_data.update({
    #                     'imgs': init_stats['imgs'],
    #                     'anns': init_stats['anns'],
    #                     'cats': init_stats['cats'].copy(),
    #                     'img_in_ann': init_stats['img_in_ann'],
    #                     'img_in_folder': init_stats.get('img_in_folder', 0)
    #                 })
    #             except Exception as e:
    #                 logger.error(f"Failed to get initial statistics: {e}")
            
    #         # Merge other datasets
    #         for other in others:
    #             try:
    #                 # Get current dataset statistics
    #                 other_stats = other.static()
    #                 logger.debug(f"Merging dataset: {other.cfg.ROOT} with {other_stats['imgs']} images, "
    #                           f"{other_stats['anns']} annotations, {len(other_stats['cats'])} categories")
                    
    #                 # Execute merge
    #                 obj._merge(other=other, cat_keep=cat_keep, overwrite=overwrite)
    #                 obj.cfg.IMGDIR_SRC = other.cfg.ROOT.joinpath(other.cfg.IMGDIR)
    #                 if save_img:    
    #                     obj.save_img()
                    
    #                 # Update statistics
    #                 static_data['imgs'] += other_stats['imgs']
    #                 static_data['anns'] += other_stats['anns']
    #                 static_data['img_in_ann'] += other_stats['img_in_ann']
    #                 static_data['img_in_folder'] += other_stats.get('img_in_folder', 0)
                    
    #                 # Merge category statistics
    #                 for cat_name, count in other_stats['cats'].items():
    #                     static_data['cats'][cat_name] = static_data['cats'].get(cat_name, 0) + count

    #             except Exception as e:
    #                 logger.error(f"Failed to merge dataset {other.cfg.ROOT}: {e}")
    #                 continue

    #         logger.info(f"Total after merging: {static_data['imgs']} images, {static_data['anns']} annotations, "
    #                   f"{len(static_data['cats'])} categories, {static_data['img_in_ann']} annotated images")
            
    #         # Validate final merge results
    #         try:
    #             final_stats = obj.static()
    #             logger.info(f"Final merged dataset: {final_stats['imgs']} images, {final_stats['anns']} annotations, "
    #                       f"{len(final_stats['cats'])} categories, {final_stats['img_in_ann']} annotated images")
    #         except Exception as e:
    #             logger.error(f"Failed to get final statistics: {e}")
    #         obj._updateIndex()
    #         return obj
            
    #     except Exception as e:
    #         logger.error(f"Error occurred during dataset merge: {e}")
    #         return None

    # def _filter(self, catIds: Optional[List[int]] = [], imgIds: Optional[List[int]] = [], annIds: Optional[List[int]] = [], mod:Optional[str]="and")->List[int]:
    #     """
    #     过滤数据集
    #     mod: 过滤方式, "and": 同时满足, "or": 满足其一的并集
    #     """
    #     if mod == "and":
    #         return self._filter_and(catIds=catIds,imgIds=imgIds,annIds=annIds)
    #     elif mod == "or":
    #         return self._filter_or(catIds=catIds,imgIds=imgIds,annIds=annIds)
    #     else:
    #         raise ValueError(f"Invalid mod: {mod}")

    
    # def _filter_and(self, catIds: Optional[List[int]] = [], imgIds: Optional[List[int]] = [], annIds: Optional[List[int]] = [])->List[int]:
    #     """
    #     根据条件过滤信息获取标注id,同时对于为[]的条件，则表示忽略
    #     """
    #     # 获取所有标注id
    #     final_annIds = self.data.getAnnIds()
        
    #     # 如果指定了图片id,则过滤
    #     if imgIds:
    #         img_annIds = self.data.getAnnIds(imgIds=imgIds)
    #         final_annIds = [ann for ann in final_annIds if ann in img_annIds]
            
    #     # 如果指定了类别id,则过滤    
    #     if catIds:
    #         cat_annIds = self.data.getAnnIds(catIds=catIds)
    #         final_annIds = [ann for ann in final_annIds if ann in cat_annIds]
            
    #     # 如果指定了标注id,则过滤
    #     if annIds:
    #         final_annIds = [ann for ann in final_annIds if ann in annIds]
            
    #     return final_annIds
    
    # def _filter_or(self, catIds: Optional[List[int]] = [], imgIds: Optional[List[int]] = [], annIds: Optional[List[int]] = [])->List[int]:
    #     """
    #     根据条件过滤信息获取标注id,满足任一条件的并集
    #     为空的条件表示不获取任何值
    #     """
    #     # 分别获取满足每个条件的标注id
    #     res_cats = self._filter_and(catIds=catIds) if catIds else []  # 满足类别条件的标注id
    #     res_imgs = self._filter_and(imgIds=imgIds) if imgIds else []  # 满足图片条件的标注id 
    #     res_anns = self._filter_and(annIds=annIds) if annIds else []  # 满足标注id条件的标注id
        
    #     # 计算所有满足条件的标注id的并集
    #     final_annIds = set(res_cats) | set(res_imgs) | set(res_anns)
    #     return list(final_annIds)
    
    # def _get_imgIds_by_annIds(self,annIds:List[int])->List[int]:
    #     """
    #     根据annIds获取图片id
    #     """
    #     if not self.data:
    #         return []
    #     return list(set([ann['image_id'] for ann in self.data.dataset['annotations'] if ann['id'] in annIds]))
    
    def _get_catIds_by_annIds(self,annIds:List[int])->List[int]:
        """
        根据annIds获取类别id
        """
        if not self.data:
            return []
        return list(set([ann['category_id'] for ann in self.data.dataset['annotations'] if ann['id'] in annIds]))
    
    def _get_data(self,annIds:List[int],level:str="img")->Tuple[List[int],List[int],List[int]]:
        """
        根据annIds获取数据
        """
        if not annIds or not self.data:
            return [],[],[]
        
        if level == "img":
            res_imgIds = self._get_imgIds_by_annIds(annIds=annIds)
            res_annIds = self.data.getAnnIds(imgIds=res_imgIds)
            res_catIds = self._get_catIds_by_annIds(annIds=res_annIds)
        elif level == "ann":
            res_imgIds = self._get_imgIds_by_annIds(annIds=annIds)
            res_annIds = annIds
            res_catIds = self._get_catIds_by_annIds(annIds=res_annIds)
        else:
            raise ValueError(f"Invalid level: {level}")
        
        return res_catIds,res_imgIds,res_annIds
    
    # def _gen_dict(self,catIds:List[int],imgIds:List[int],annIds:List[int],alignCat:bool=True,keep_all_img:bool=True)->dict:
    #     if not keep_all_img and (not catIds or not self.data):
    #         return {}
    #     new_dataset = {
    #         'info': self.data.dataset.get('info',[]),
    #         'licenses': self.data.dataset.get('licenses',[]),
    #         'images': [img for img in self.data.dataset['images'] if img['id'] in imgIds],
    #         'annotations': [ann for ann in self.data.dataset['annotations'] if ann['id'] in annIds],
    #         'categories': [cat for cat in self.data.dataset['categories'] if cat['id'] in catIds]
    #     }
    #     if isinstance(alignCat,bool):
    #         new_dataset['categories'] = [cat for cat in self.data.dataset['categories']]

    #     return new_dataset

    # def filter(self, 
    #            cats: Optional[List[Union[int,str]]] = [], 
    #            imgs: Optional[List[Union[int,str]]] = [], 
    #            annIds: Optional[List[int]] = [], 
    #            mod:Optional[str]="and", 
    #            level:str="img",
    #            revert:bool=False,
    #            dst_file:Optional[COCOX]=None,
    #            alignCat:bool=True,
    #            keep_all_img:bool=False, # 会将所有的数据保留下来
    #            )->Union[COCOX,None]:
    #     """
    #     过滤数据集,支持多种过滤方式:
    #     - 图像: 支持id或文件名模糊搜索
    #     - 类别: 支持id或名称搜索
    #     - 标注: 支持标注id过滤

    #     参数:
    #         cats (List[Union[int,str]], optional): 类别id或名称列表,默认为空列表
    #         imgs (List[Union[int,str]], optional): 图像id或文件名列表,默认为空列表
    #         annIds (List[int], optional): 标注id列表,默认为空列表
    #         mod (str, optional): 过滤模式,"and"表示同时满足所有条件,"or"表示满足任一条件,默认为"and"
    #         level (str, optional): 过滤级别,"img"表示按图片过滤,"ann"表示按标注过滤,默认为"img"
    #         revert (bool, optional): 是否反向过滤,默认为False
    #         dst_file (Optional[COCOX], optional): 目标COCOX对象,用于指定保存路径,默认为None
    #         alignCat (bool, optional): 是否对齐类别,True表示保留所有类别,默认为True
    #         keep_all_img (bool, optional): 是否保留所有图片,True表示保留所有图片但只保留符合条件的标注,默认为True

    #     返回:
    #         Union[dict,COCOX]: 返回过滤后的COCOX对象
            
    #     TODO: 增加未标注数据作为负样本，特别是公式检测，需要设置数量或则比例
    #     """
               
    #     # 获取id
    #     imgIds = [self._get_img(img,return_id=True)[1] for img in imgs if self._get_img(img,return_id=True)[0]]
    #     catIds = [self._get_cat(cat,return_id=True)[1] for cat in cats if self._get_cat(cat,return_id=True)[0]]
        
    #     dst_annIds = self._filter(catIds=catIds, imgIds=imgIds, annIds=annIds,mod=mod)
        
    #     # 如果过滤后没有数据，返回空字典或空对象
    #     if not dst_annIds and not keep_all_img:
    #         return COCOX(data=None, cfg=dst_file if dst_file else None)
        
    #     # 创建新对象,并更新配置
    #     obj = COCOX(data=self, cfg=dst_file if dst_file else None)
    #     if dst_file:
    #         obj.cfg = dst_file
            
    #     if revert:
    #         if level == "ann":
    #             all_annIds = [ann['id'] for ann in obj.data.dataset['annotations']] 
    #             dst_annIds = set(all_annIds) - set(dst_annIds)
    #         elif level == "img":
    #             # 获取图片id 
    #             _,dst_imgIds,_ = obj._get_data(annIds=dst_annIds,level=level)
    #             all_imgIds = [img['id'] for img in obj.data.dataset['images']] 
    #             dst_imgIds = set(all_imgIds) - set(dst_imgIds)
    #             dst_annIds = self._filter(imgIds=dst_imgIds)
                
    #     dst_catIds,dst_imgIds,dst_annIds = obj._get_data(annIds=dst_annIds,level=level)
        
    #     # 如果keep_all_img为True,保留所有图片
    #     if keep_all_img:
    #         dst_imgIds = [img['id'] for img in obj.data.dataset['images']]
            
    #     dst_dict = obj._gen_dict(catIds=dst_catIds,imgIds=dst_imgIds,annIds=dst_annIds,alignCat=alignCat)
    #     # 将coco字典转换为对象
    #     obj.data = obj.dict2_(dst_dict)
        
    #     static_data = obj.static()
    #     logger.info(f"{obj.cfg.ROOT.joinpath(obj.cfg.ANNDIR).joinpath(obj.cfg.ANNFILE)}[{type(obj.data).__name__ if obj.data else 'None'}]: Images:{static_data['imgs']}, Annotations:{static_data['anns']}, Categories:{len(static_data['cats'])}")
    #     return obj
       
 
    # def correct(self, api_url:Callable, cats:Union[int,str,list], dst_file:Optional[CCX]=None):
    #     """
    #     使用API纠正标注类别
    #     """
    #     obj = COCOX(data=self, cfg=dst_file if dst_file else None)
        
    #     if isinstance(cats,list):
    #         catIds = [obj._get_cat(cat,return_id=True)[1] for cat in cats]
    #     else:
    #         catIds = [obj._get_cat(cats,return_id=True)[1]]
                     
    #     sys_anns = obj.data.getCatIds()
    #     imgIds = obj.data.getImgIds()

    #     for img_id in imgIds:
    #         img = obj.data.loadImgs(img_id)[0]
    #         imgpath = obj.cfg.IMGDIR_SRC.joinpath(img['file_name'])

    #         one_annoIds = obj.data.getAnnIds(imgIds=img['id'])
    #         one_anns = obj.data.loadAnns(one_annoIds)
    #         for ann in one_anns:                
    #             if ann['category_id'] not in catIds:
    #                 continue
                
    #             bbox = ann['bbox']
    #             x, y, w, h = bbox
    #             x1, y1 = int(x), int(y)
    #             x2, y2 = int(x+w), int(y+h)
    #             bbox = [x1,y1,x2,y2]

    #             # Call API, pass in the temporary file path
    #             api_res = api_url(imgpath,bbox)
    #             if api_res is None:
    #                 api_res = ann['category_id']
    #             if api_res not in sys_anns:
    #                 raise ValueError(f"API returned category ID is not in system categories, please confirm the value is between 1 and {len(sys_anns)}")
    #             # Apply API results directly to annotation
    #             ann['category_id'] = api_res

    #     return obj

    # @staticmethod
    # def _split(imglists:List[str],ratio:Union[List[float],int]=[0.7,0.2,0.1],by_file=False):
    #     # 检查ratio是否为整数
    #     if isinstance(ratio, int):
    #         # 如果是整数,生成等比例的ratio列表
    #         ratio = [1/ratio] * ratio
            
    #     # 确保ratio之和为1
    #     if abs(sum(ratio) - 1) > 0.0001:
    #         ratio = [r/sum(ratio) for r in ratio]
            
    #     if by_file:
    #         # 按文件名分组
    #         samebooks = defaultdict(list)
    #         for image in imglists:
    #             match = re.match(r'(.+?)_(\d+)\..*', image)
    #             if match:
    #                 prefix, _ = match.groups()
    #                 samebooks[prefix].append(image)
            
    #         # 对每个文件组进行划分
    #         split_data = defaultdict(list)
    #         file_list = list(samebooks.keys())
            
    #         for file in file_list:
    #             random.shuffle(samebooks[file])
    #             start = 0
    #             last_index = 0
    #             for i,r in enumerate(ratio):
    #                 end = start + int(len(samebooks[file])*r)
    #                 split_data[i].extend(samebooks[file][start:end])
    #                 start = end
    #                 last_index = i
    #             split_data[last_index].extend(samebooks[file][start:])
            
    #         return [split_data[i] for i in range(len(ratio))]
            
    #     else:
    #         # 直接对图片列表进行随机划分
    #         random.shuffle(imglists)
    #         total = len(imglists)
    #         split_data = []
    #         start = 0
    #         for r in ratio:
    #             end = start + int(total * r)
    #             split_data.append(imglists[start:end])
    #             start = end
    #         split_data[-1].extend(imglists[start:])  # 将剩余的图片添加到最后一组
            
    #         return split_data    


    # def split2(self,ratio:List[float]=[0.7,0.2,0.1],by_file=False,dst_file:Optional[CCX]=None,merge:bool=True)->List[COCOX]:
    #     """
    #     将数据集按照指定比例划为训练集、验证集和测试集

    #     Args:
    #         ratio (List[float]): 数据集划分比例，按照[训练集,验证集,测试集]顺序,默认[0.7,0.2,0.1]
    #         by_file (bool): 是否按照PDF文件名进行划分,True则同一PDF的页面会被分到同一数据集,False则完全随机划分
    #         newObj (Optional[COCOX]): 新的COCOX对象,用于保存划分后的数据集,实际只有ROOT有用，其他参数无效
    #         visual (bool): 是否可视化显示划分结果，同时会保存数据
    #         merge (bool): 是否将划分后的图像合并到同一文件夹下

    #     Returns:
    #         Union[Tuple[COCOX,COCOX,COCOX],Tuple[dict,dict,dict]]: 
    #         如果merge=True,返回三个COCOX对象,分别对应训练集、验证集、测试集
    #         如果merge=False,返回三个字典,包含各自数据集的图像信息
    #     """
    #     # 获取各集合的图片ID和标注ID
    #     def get_ids(img_set):
    #         if not img_set:
    #             return [], []
    #         img_ids = [obj._get_img(img,return_id=True)[1] for img in img_set]
    #         ann_ids = obj.data.getAnnIds(imgIds=img_ids)
    #         return img_ids, ann_ids
        
    #     # 生成数据字典并创建对象
    #     def create_split_obj(cat_ids, img_ids, ann_ids, split_type):
    #         split_dict = obj._gen_dict(catIds=cat_ids, imgIds=img_ids, annIds=ann_ids)
    #         split_obj = copy.deepcopy(obj)
    #         split_obj.cfg.ANNFILE = f"instances_{split_type}.json"
    #         if merge:
    #             split_obj.cfg.IMGDIR = f"{split_type}"
    #         split_obj.data = split_obj.dict2_(split_dict)
    #         return split_obj
        
        
    #     obj = COCOX(data=self, cfg=dst_file if dst_file else None)
        
    #     imglists = obj._get_imglist()
    #     split_sets = obj._split(imglists=imglists,ratio=ratio,by_file=by_file)  
        
    #     catIds = list(range(1,len(obj.data.dataset['categories'])+1))
    #     all_split_objs = []
        
    #     if len(ratio) == 2:
    #         split_types = ["train","val"]
    #     elif len(ratio) == 3:
    #         split_types = ["train","val","test"]
    #     else:
    #         split_types = [f"split_{i}" for i in range(len(ratio))]
            
    #     for part_set,split_type in zip(split_sets,split_types):
    #         imgIds,annIds = get_ids(part_set)
    #         split_obj = create_split_obj(catIds,imgIds,annIds,split_type)
    #         split_obj._updateIndex()
    #         all_split_objs.append(split_obj)
            
    #     return all_split_objs,split_types
              
    # def split(self,ratio:List[float]=[0.7,0.2,0.1],by_file=False,dst_file:Optional[CCX]=None,merge:bool=True)->Union[Tuple[COCOX,COCOX,COCOX],Tuple[dict,dict,dict]]:
    #     """
    #     将数据集按照指定比例划为训练集、验证集和测试集

    #     Args:
    #         ratio (List[float]): 数据集划分比例，按照[训练集,验证集,测试集]顺序,默认[0.7,0.2,0.1]
    #         by_file (bool): 是否按照PDF文件名进行划分,True则同一PDF的页面会被分到同一数据集,False则完全随机划分
    #         newObj (Optional[COCOX]): 新的COCOX对象,用于保存划分后的数据集,实际只有ROOT有用，其他参数无效
    #         visual (bool): 是否可视化显示划分结果，同时会保存数据
    #         merge (bool): 是否将划分后的图像合并到同一文件夹下

    #     Returns:
    #         Union[Tuple[COCOX,COCOX,COCOX],Tuple[dict,dict,dict]]: 
    #         如果merge=True,返回三个COCOX对象,分别对应训练集、验证集、测试集
    #         如果merge=False,返回三个字典,包含各自数据集的图像信息
    #     """
    #     obj = COCOX(data=self, cfg=dst_file if dst_file else None)
        
    #     imglists = obj._get_imglist()
    #     total_imgs = len(imglists)

    #     if by_file:
    #         samebooks = defaultdict(list)
    #         for image in imglists:
    #             match = re.match(r'(.+?)_(\d+)\..*', image)
    #             if match:
    #                 prefix, page = match.groups()
    #                 samebooks[prefix].append(image)
    #         samebooks = dict(samebooks)
            
    #         # 对每个key下的数据进行划分
    #         split_data = {}
    #         for key, images in samebooks.items():
    #             total = len(images)
    #             train_end = int(total * ratio[0])
                
    #             if len(ratio) == 2:
    #                 val_end = total  # 0.3包含剩余所有数据
    #             else:
    #                 val_end = int(total * (ratio[0] + ratio[1]))
                
    #             # 随机打乱图片列表
    #             random.shuffle(images)
                
    #             # 划分数据
    #             split_data[key] = {
    #                 'train': images[:train_end],
    #                 'val': images[train_end:val_end],
    #                 'test': images[val_end:] if len(ratio) > 2 else []
    #             }
            
    #         # 合并所有子列表
    #         train_set = [img for key in split_data for img in split_data[key]['train']]
    #         val_set = [img for key in split_data for img in split_data[key]['val']]
    #         test_set = [img for key in split_data for img in split_data[key]['test']] if len(ratio) > 2 else []
    #     else:
    #         # 随机打乱图片列表
    #         random.shuffle(imglists)
            
    #         train_end = int(total_imgs * ratio[0])
            
    #         if len(ratio) == 2:
    #             # 两个数据集情况,val_set包含余所有数据
    #             train_set = imglists[:train_end]
    #             val_set = imglists[train_end:]
    #             test_set = []
    #         else:
    #             # 三个数据集情况,test_set包含剩余所有数据
    #             val_end = int(total_imgs * (ratio[0] + ratio[1]))
    #             train_set = imglists[:train_end]
    #             val_set = imglists[train_end:val_end]
    #             test_set = imglists[val_end:]
        
        
    #     # 获取各集合的图片ID和标注ID
    #     def get_ids(img_set):
    #         if not img_set:
    #             return [], []
    #         img_ids = [obj._get_img(img,return_id=True)[1] for img in img_set]
    #         ann_ids = obj.data.getAnnIds(imgIds=img_ids)
    #         return img_ids, ann_ids
            
    #     train_imgIds, train_annIds = get_ids(train_set)
    #     val_imgIds, val_annIds = get_ids(val_set) 
    #     test_imgIds, test_annIds = get_ids(test_set)
        
    #     # 所有集合共用相同的类别ID
    #     cat_ids = list(range(1,len(obj.data.dataset['categories'])+1))
        
    #     # 生成数据字典并创建对象
    #     def create_split_obj(img_ids, ann_ids, split_type):
    #         split_dict = obj._gen_dict(catIds=cat_ids, imgIds=img_ids, annIds=ann_ids)
    #         split_obj = copy.deepcopy(obj)
    #         split_obj.cfg.ANNFILE = getattr(split_obj.cfg, f'ANNFILE_{split_type.upper()}')
    #         if merge:
    #             split_obj.cfg.IMGDIR = getattr(split_obj.cfg, f'IMGDIR_{split_type.upper()}')
    #         split_obj.data = split_obj.dict2_(split_dict)
    #         return split_obj
            
    #     train_obj = create_split_obj(train_imgIds, train_annIds, 'val')
    #     val_obj = create_split_obj(val_imgIds, val_annIds, 'val')
    #     test_obj = create_split_obj(test_imgIds, test_annIds, 'test')

    #     train_obj._updateIndex()
    #     val_obj._updateIndex()
    #     test_obj._updateIndex()
    #     return train_obj, val_obj, test_obj

    # def img2coco(self,imgpath:str, img_id:int, dst_file:CCX=None):
    #     """
    #     将图片转换为coco格式
    #     """
    #     if not self.data:
    #         return None
        
    #     if not os.path.exists(imgpath):
    #         raise FileNotFoundError(f"Image file {imgpath} not found")
        
    #     # 创建新的数据集对象
    #     new_data = {
    #         'info': self.data.dataset.get('info', []),
    #         'licenses': self.data.dataset.get('licenses', []),
    #         'images': [],
    #         'annotations': [],
    #         'categories': self.data.dataset.get('categories', [])
    #     }
        
    #     # 添加图片信息
    #     img_info = {
    #         'id': img_id,
    #         'file_name': os.path.basename(imgpath),
    #         'width': 0,
    #         'height': 0
    #     }
    #     new_data['images'].append(img_info)
        
    #     # 添加标注信息
    #     for ann in self.data.dataset['annotations']:
    #         if ann['image_id'] == img_id:
    #             new_data['annotations'].append(ann)
        
    #     # 创建新的COCOX对象
    #     new_obj = COCOX(data=new_data, cfg=dst_file if dst_file else None)
        
    #     return new_obj

