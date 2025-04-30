from cocox import COCOX 
from pathlib import Path


if __name__ == "__main__":
    coco_path0 = "tests/data/test01/annotations/instances_default.json"
    co2x0 = COCOX(coco_path0)
    static_data = co2x0.vis_anno_info(save_dir=Path("tests/data/test01/vis"))

# 