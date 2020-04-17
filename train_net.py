#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import pickle
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.structures import BoxMode

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("tianchi_train",)
    cfg.DATASETS.TEST = ("tianchi_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # 从 Model Zoo 中获取预训练模型
    cfg.MODEL.WEIGHTS = "model/model_final_280758.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.BASE_LR = 0.00025  # 学习率
    cfg.SOLVER.MAX_ITER = 20000  # 最大迭代次数 150000/8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别：红绿灯
    cfg.NUM_GPUS = 2
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_image_dicts(img_dir, pkl_dir='/opt/gitserial/tianchi/cache'):
    # image_annotation.pkl
    image_annotation_dict = pickle.load(open(os.path.join(pkl_dir, img_dir + "_image_annotation.pkl"), 'rb'))
    # with open(json_file) as f:
    #     imgs_anns = json.load(f)
    dataset_dicts = []
    idx = 0
    for _, vv in enumerate(image_annotation_dict.values()):
        for v in vv:
            record = {}
            # filename = os.path.join(img_dir, v["file_name"])
            filename = v['file_name']
            # height, width = cv2.imread(filename).shape[:2]
            height, width = v['height'], v['width']

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            # 096837
            # {'file_name': '/opt/gitserial/tianchi/data/train_dataset_part1/image/096837/0.jpg',
            #  'height': 800,
            #  'width': 800,
            #  'annotation': {'item_id': '096837',
            #                 'img_name': '0.jpg',
            #                 'annotations': [{'label': '短袖连衣裙',
            #                                  'box': [40, 124, 248, 576],
            #                                  'display': 1,
            #                                  'viewpoint': 0,
            #                                  'instance_id': 29683701},
            #                                 {'label': '短袖连衣裙',
            #                                  'box': [275, 107, 491, 558],
            #                                  'display': 1,
            #                                  'viewpoint': 0,
            #                                  'instance_id': 29683701},
            #                                 {'label': '短袖连衣裙',
            #                                  'box': [542, 113, 780, 561],
            #                                  'display': 1,
            #                                  'viewpoint': 0,
            #                                  'instance_id': 29683701}]},
            #  'categories': {'id': 12, 'name': '短袖连衣裙'}}
            annos = v['annotation']
            objs = []
            for anno in annos['annotations']:
                # assert not anno["region_attributes"]
                # anno = anno["shape_attributes"]
                # px = anno["all_points_x"]
                # py = anno["all_points_y"]
                # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                # poly = [p for x in poly for p in x]

                obj = {
                    # "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox": [*anno['box']],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            idx += 1
    return dataset_dicts


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    for d in ["train", "val"]:
        DatasetCatalog.register("tianchi_" + d, lambda d=d: get_image_dicts(d))
        MetadataCatalog.get("tianchi_" + d).set(thing_classes=["cloth"])
        if d == 'val':
            MetadataCatalog.get("tianchi_val").evaluator_type = "coco"
    # balloon_metadata = MetadataCatalog.get("tianchi_train")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    # cd tools /
    # ./train_net.py --num-gpu 2 --resume --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
