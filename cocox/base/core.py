# import faster_coco_eval

# faster_coco_eval.init_as_pycocotools()

from pycocotools.coco import COCO


from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import json

class _COCO(COCO):
    def __init__(self, annotation_file=None):
        # super().__init__(annotation_file)
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self.createIndex()
    
    def createIndex(self):
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])


        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
       
    def showBBox(self, anns, label_box=True,
                 colors=[
                    (1.0, 0.0, 0.0),    # 红色
                    (0.0, 1.0, 0.0),    # 绿色
                    (0.0, 0.0, 1.0),    # 蓝色
                    (1.0, 1.0, 0.0),    # 黄色
                    (1.0, 0.0, 1.0),    # 品红
                    (0.0, 1.0, 1.0),    # 青色
                    (1.0, 0.5, 0.0),    # 橙色
                    (0.5, 0.0, 1.0),    # 紫色
                    (0.0, 0.5, 0.5),    # 青绿色
                    (0.5, 0.5, 0.0),    # 橄榄色
                    (1.0, 0.0, 0.5),    # 玫瑰红
                    (0.0, 1.0, 0.5),    # 春绿
                    (0.5, 0.0, 0.0),    # 栗色
                    (0.0, 0.5, 1.0),    # 天蓝色
                    (1.0, 0.75, 0.8),   # 粉色
                ]):
        """
        show bounding box of annotations or predictions
        anns: loadAnns() annotations or predictions subject to coco results format
        label_box: show background of category labels or not
        """
        if len(anns) == 0:
            return 0
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        image2color=dict()
        for i, cat in enumerate(self.getCatIds()):
            if i < len(colors):
                image2color[cat] = colors[i]
            else:
                image2color[cat] = (np.random.random((1, 3))*0.7+0.3).tolist()[0]
        for ann in anns:
            c=image2color[ann['category_id']]
            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            color.append(c)
            # option for dash-line
            # ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=c, linewidth=2))
            fontsize=4
            if label_box:
                label_bbox=dict(facecolor=c, pad=0,edgecolor=c)
                
            else:
                label_bbox=None
            if 'score' in ann:
                ax.text(bbox_x, bbox_y, '%s: %.2f'%(self.loadCats(ann['category_id'])[0]['name'], ann['score']), color='white', bbox=label_bbox,fontsize=fontsize)
            else:
                ax.text(bbox_x, bbox_y, '%s'%(self.loadCats(ann['category_id'])[0]['name']), color='white', bbox=label_bbox,fontsize=fontsize)
        # option for filling bounding box
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        # p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)


