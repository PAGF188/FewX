import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

#Cambiamos a FSOD dataset. Tiene que estar en ./datasets

# ==== Predefined datasets and splits for fsod ==========

# fsod
#   - annotations
#       - instances_train2017.json
#       - instances_val2017.json
#   - images

#______["NOMBRE_DATASET"] = {
#    "NOMBRE REGISTRO (puede ser usado en .conf)":(),
#    "NOMBRE REGISTRO": ("image_root", ".json file"),
#}

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["fsod"] = {
    "fsod_train": ("fsod/images", "fsod/annotations/instances_train2017.json"),
    "fsod_eval": ("fsod/images", "fsod/annotations/instances_val2017.json"),
}

def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        print(dataset_name)
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
