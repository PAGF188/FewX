from pycocotools.coco import COCO
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import math
import os
import pandas as pd
import json
import shutil

######################################
annFile = './annotations/instances_val2017.json'
shot=5
#######################################
dump = '{"images": ['
coco = COCO(annFile)
n_classes = len(coco.getCatIds())
print("Numero de clases:", n_classes)

#Tomamos 'shot' clases de cada categoria y cogemos sus anotaciones
ann_ids_final = []
for cat in coco.cats:
    ann_ids = coco.getAnnIds(catIds=[cat])
    ann_ids_final = ann_ids_final + ann_ids[0:shot]

annotations=coco.loadAnns(ids=ann_ids_final)
print(len(annotations))
#Obtenemos el id de las imágenes de annotations
img_ids = []
for i in range(n_classes*shot):
    img_ids.append(annotations[i]['image_id'])

print("Numero de imagenes: ", len(img_ids))

#Obtenemos resto de información (file_name,etc....)
images = coco.loadImgs(img_ids)
for im in images:
    dump = dump + str(im) + ','
dump = dump[:-1]
dump = dump + '],"type": "instances", "annotations": '
dump = dump + str(annotations) + ', "categories": '
dump = dump + str(coco.loadCats(coco.getCatIds())) + '}'
dump = dump.replace('\'',"\"")
print(dump)
filename = './final_split_' + str(shot) + '_shot.json'

with open(filename, "w") as f:
    f.write(dump)

