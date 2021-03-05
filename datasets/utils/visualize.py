from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import random
from detectron2.utils.visualizer import Visualizer
import cv2
from PIL import Image
import numpy as np


register_coco_instances("fsod", {}, "annotations/instances_train2017.json", "images/")
m = MetadataCatalog.get("fsod")
print(m)
dataset_dicts = DatasetCatalog.get("fsod")

for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=m, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    img = Image.fromarray(vis.get_image()[:, :, ::-1], 'RGB')
    img.show()
