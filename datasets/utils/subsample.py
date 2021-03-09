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
annFile = './annotations/instances_val2017.json_reales'
nclases=5  # con clases 1,2,3.....,200
#######################################

dump = '{"images": ['
coco = COCO(annFile)

ann_ids_final = []
for cat in range(nclases+1):
    ann_ids = coco.getAnnIds(catIds=[cat])
    ann_ids_final = ann_ids_final + ann_ids[:]

annotations=coco.loadAnns(ids=ann_ids_final)
#print(annotations)

#Obtenemos el id de las imágenes de annotations
img_ids = []
for i in range(len(annotations)):
    img_ids.append(annotations[i]['image_id'])

print("Numero de imagenes: ", len(img_ids))

images = coco.loadImgs(img_ids)
for im in images:
    dump = dump + str(im) + ','
dump = dump[:-1]
dump = dump + '],"type": "instances", "annotations": '
dump = dump + str(annotations) + ', "categories": '

dump = dump + str(coco.loadCats(coco.getCatIds(catIds=list(range(1,nclases+1))))) + '}'
dump = dump.replace('\'',"\"")
#print(dump)
filename = './instances_val2017.json'

with open(filename, "w") as f:
    f.write(dump)
