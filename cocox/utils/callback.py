from .logger import logger
from .common import IMG_EXT
def callback_get_input(img_input, ann_input, **kwargs):
    logger.info(f"img_input: {img_input}")
    logger.info(f"ann_input: {ann_input}")
    return (img_input, ann_input)

def callback_process_cat(img_input, ann_input, **kwargs):
    import cv2
    process_cat = kwargs.pop('process_cat',lambda x,y,z: z)
    img = cv2.imread(img_input['imgpath'])
    cat_map = img_input['cats'] # int -> str
    for id, ann in ann_input.items():
        category_id = ann['category_id']
        bbox = ann['bbox']
        new_category_id = process_cat(img, bbox,category_id,**kwargs)
        if new_category_id is not None and new_category_id in cat_map.keys():
            ann_input[id]['category_id'] = new_category_id
    
    return (img_input, ann_input)
    

def callback_process_img_name(img_input, ann_input, **kwargs):
    # 根据给出的数据进行文件名称修改，仅仅修改名称，不涉及到图片以及标注修改
    import os
    from pathlib import Path
    import shutil
    # 获取参数
    src_imgs = kwargs.pop('src_imgs', [])  # 目标图片列表，不存在则使用空列表
    dst_imgs = kwargs.pop('dst_imgs', [])  # 目标图片列表，不存在则使用空列表
    global_info = kwargs.pop('global_info', {})  # 全局信息，不存在则使用空字典

    if len(src_imgs) != len(dst_imgs):
        logger.warning("src_imgs和dst_imgs的长度不一致")
        return None
        
    # 获取原始图片信息
    file_name = img_input['file_name']  # 原始图片路径
    # 图片完整路径
    img_path = img_input['imgpath']
        
    # 使用zip方式高效匹配源图片和目标图片
    for src_img, dst_img in zip(src_imgs, dst_imgs):
        if Path(src_img).name == file_name:
            img_input['file_name'] = dst_img
            src_imgs.remove(src_img)
            dst_imgs.remove(dst_img)
            dst_path = img_input['root'].joinpath(img_input['imgdir']).joinpath(img_input['imgfolder']).joinpath(dst_img)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # 如果源文件和目标文件在同一目录，则使用重命名而不是复制
            if img_path.parent == dst_path.parent:
                os.rename(img_path, dst_path)
                break
            # 否则复制文件
            shutil.copy2(img_path, dst_path)
            break
        else:
            if 'missing_imgs' not in global_info:
                global_info['missing_imgs'] = []
            if 'missing_imgs_num' not in global_info:
                global_info['missing_imgs_num'] = 0
            global_info['missing_imgs'].append(file_name)
            global_info['missing_imgs_num'] += 1
    
    return (img_input, ann_input)

def callback_process_img_size(img_input, ann_input, **kwargs):
    """
    修改图片信息以及对应的标注框信息，根据目标图片列表中的图片尺寸调整原始标注数据。
    
    该函数会查找与原始图片匹配的目标图片，并根据目标图片的尺寸调整标注数据。
    如果找到匹配的图片，会将其复制或移动到指定目录，并按照新图片的尺寸比例调整所有标注信息。
    
    参数:
        img_input (dict): 包含图片信息的字典，包括路径、宽高等
        ann_input (dict): 包含标注信息的字典，键为标注ID，值为标注数据
        **kwargs: 额外参数
            - img_list (list): 目标图片路径列表，必须提供
            - strict (bool): 是否严格匹配文件名（包括扩展名），默认为False
            - file_mode (str): 文件操作模式，'copy'或'move'，默认为'copy'
            - global_info (dict): 用于记录处理过程中的全局信息
    
    返回:
        tuple: 更新后的(img_input, ann_input)，如果未找到匹配图片则返回原始数据
    """
    import cv2
    import os
    from pathlib import Path
    import shutil
    
    # 获取参数
    img_list = kwargs.pop('img_list', [])  # 目标图片列表，不存在则使用空列表
    strict = kwargs.pop('strict', False)   # 是否严格匹配文件名，默认为False
    global_info = kwargs.pop('global_info', {})  # 全局信息，不存在则使用空字典
    file_mode = kwargs.pop('file_mode', 'copy')  # 文件操作模式，默认为'copy'
    # 获取原始图片信息
    img_path = img_input['imgpath']  # 原始图片路径
    root = img_input['root']         # 根目录
    
    # 提取相对路径（images目录之后的部分）
    img_path_str = str(img_path)
    img_rel_path = ""
    
    if "images" in img_path_str:
        parts = img_path_str.split("images")
        if len(parts) > 1:
            img_rel_path = parts[1][1:]
        else:
            logger.warning("Unable to extract the part after 'images' from the path")
    else:
        logger.warning("Path does not contain 'images' directory")
    
    # 构建目标图片路径
    dst_imgpath = img_input['root'].joinpath(img_input['imgdir']) / img_input['imgfolder'] / img_rel_path
    img_name = Path(img_path).name
    
    # 查找匹配的目标图片
    img_name_without_ext = os.path.splitext(img_name)[0]
    found_img_path = None
    
    for dst_img_path in img_list:
        dst_img_name = os.path.basename(dst_img_path)
        dst_img_name_without_ext = os.path.splitext(dst_img_name)[0]
        
        # 根据strict参数决定匹配方式
        if strict:
            # 严格匹配：文件名必须完全相同（包括扩展名）
            if img_name == os.path.basename(dst_img_path):
                found_img_path = Path(dst_img_path)
                break
        else:
            # 非严格匹配：只比较不含后缀的文件名
            if img_name_without_ext == dst_img_name_without_ext:
                found_img_path = Path(dst_img_path)
                break
    
    # 如果没有找到匹配的图片，返回原始数据
    if found_img_path is None:
        if 'missing_imgs' not in global_info:
                global_info['missing_imgs'] = []
        if 'missing_imgs_num' not in global_info:
            global_info['missing_imgs_num'] = 0
        global_info['missing_imgs'].append(img_name)
        global_info['missing_imgs_num'] += 1
        return img_input, ann_input
    
    # 最终移动到的图像目录
    if img_input['root'].joinpath(img_input['imgdir']).exists():
        final_dst_imgpath = img_input['root'].joinpath("correct_images") / img_input['imgfolder'] / found_img_path.name
    else:
        final_dst_imgpath = dst_imgpath


    # 复制找到的图片到目标目录
    final_dst_imgpath.parent.mkdir(parents=True, exist_ok=True)
    # 根据file_mode参数决定是移动还是复制图片
    if file_mode == 'move':
        # 删除原始文件
        # os.remove(img_path)
        
        shutil.move(found_img_path, final_dst_imgpath)
    else:
        shutil.copy(found_img_path, final_dst_imgpath)
    img_list.remove(str(found_img_path))  # 从列表中移除已处理的图片
    
    # 读取新图片的尺寸
    new_img = cv2.imread(str(found_img_path))
    new_img_width, new_img_height = new_img.shape[1], new_img.shape[0]
    
    # 计算缩放比例
    scale_w = new_img_width / img_input['width']
    scale_h = new_img_height / img_input['height']
    
    # 更新图片信息
    img_input['width'] = new_img_width
    img_input['height'] = new_img_height
    
    # 更新图片名称
    img_input['file_name'] = found_img_path.name
    
    # 更新标注信息
    for id, ann in ann_input.items():
        # 更新边界框
        bbox = ann['bbox']
        bbox[0] = bbox[0] * scale_w  # x
        bbox[1] = bbox[1] * scale_h  # y
        bbox[2] = bbox[2] * scale_w  # width
        bbox[3] = bbox[3] * scale_h  # height
        
        # 更新面积
        ann['area'] *= scale_w * scale_h
        
        # 更新分割点
        seg = ann['segmentation']
        if seg is not None and len(seg) > 0:
            # assert len(seg) % 2 == 0, "segmentation 的长度必须是偶数"
            # 遍历seg中的每个点坐标
            for j in range(0, len(seg), 2):
                try:
                    seg[j] = float(seg[j]) * scale_w      # x坐标
                    if j+1 < len(seg):
                        seg[j+1] = float(seg[j+1]) * scale_h  # y坐标
                except Exception as e:
                    print(f"{img_input['file_name']} 的{id}标注信息有误，请检查")

    
    return (img_input, ann_input)
