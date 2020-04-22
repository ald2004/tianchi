
import logging
import os, time
import pickle

import detectron2.utils.comm as comm
import torch
import torch.nn as nn
import torch.optim as optim
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    detection_utils as utils,
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.solver import (
    build_lr_scheduler,
    build_optimizer,
)
from detectron2.structures import ImageList
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d  -   %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
DATA_ROOT_DIR = 'data'  # tcdata_train/train_dataset_part1
IS_LOCAL = True  # if local or aliyun
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class MatchModel(nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 1024, 3, padding=1)
        self.fc1 = nn.Linear(1024 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.fc1(x1.view(x1.size(0), -1))
        x1 = self.fc2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.fc1(x2.view(x1.size(0), -1))
        x2 = self.fc2(x2)
        x = x1 - x2
        x = x * x
        x = self.fc3(x)
        return x


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # args.config_file = 'configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_list(args.opts)
    # cfg.DATASETS.TRAIN = ("tianchi_train",)
    # cfg.DATASETS.TEST = ("tianchi_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    # 从 Model Zoo 中获取预训练模型
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.BASE_LR = 0.00025  # 学习率
    cfg.SOLVER.MAX_ITER = 10000  # 最大迭代次数 150000/8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别：红绿灯
    cfg.NUM_GPUS = 2

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)  # if you don't like any of the default setup, write your own setup code
    return cfg


def uni_test():
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 2  # The number of epochs of training 迭代轮数
    model = MatchModel()
    input1 = torch.randn(1, 256, 7, 7)
    input2 = torch.randn(1, 256, 7, 7)

    output = model(input1, input2)
    loss = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    input1 = input1.to(device)
    input2 = input2.to(device)
    model = model.to(device)
    label = torch.ones(1, 2).to(device)
    for e in range(500):
        y_pred = model(input1, input2)
        lo = loss(y_pred, label)

        model.zero_grad()
        lo.backward()
        optimizer.step()
    model = model.cpu()
    print(e, lo.item())
    sd = {'model': model.state_dict(), '__author__': 'yuanpu'}
    with open('/opt/gitserial/tianchi/model/mmmodel_final_280758.pkl', 'wb') as fid:
        pickle.dump(sd, fid, pickle.HIGHEST_PROTOCOL)
    # torch.save(model, '/opt/gitserial/tianchi/model/mmmodel_final_280758.pkl')

    print(output.shape, output)


def get_trainval_pos_pair_dicts(dataset_name, cache_dir='/opt/gitserial/tianchi/cache',
                                file_name='trainval_pos_pair_dict.pkl'):
    with open(os.path.join(cache_dir, file_name), 'rb') as fid:
        trainval_pos_pair_dict = pickle.load(fid)
    dataset_dicts = []
    for count, vv in enumerate(trainval_pos_pair_dict.values()):
        # print(count, '===================================')
        # if count > 4:
        #     break
        # vv ['028345', '017278', '002328'] :  {'file_pair': ['i_001504_1.jpg', 'v_001504_0', 'v_001504_120',],'train_dir_num': 'train_dataset_part6'}
        file_pair = vv['file_pair']
        train_dir_num = vv['train_dir_num']
        if not (len(file_pair)):
            continue
        x1_filename = vv['file_pair'][0]  # 'i_001504_1.jpg'
        a1, a2 = x1_filename.strip().split('_')[1], x1_filename.strip().split('_')[2]
        tmpfilename = os.path.join(os.path.abspath('.'), DATA_ROOT_DIR, train_dir_num, 'image', a1, a2)
        x2_filenames = vv['file_pair'][1:]
        if not len(x1_filename) or not len(x2_filenames):
            continue
        if not os.path.exists(tmpfilename):
            continue

        item_id = str(x1_filename.strip().split('_')[1])
        record = {}
        record["filename_x1"] = x1_filename
        record["filename_x2"] = x2_filenames
        record["train_dir_num"] = train_dir_num
        record["item_id"] = item_id
        # record["width"], record["height"] = int(800), int(800)  # dummy
        dataset_dicts.append(record)
    return dataset_dicts


def mapper(dataset_dict):
    # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # {'filename_x1': 'i_052206_3.jpg',
    #  'filename_x2': ['v_052206_0',
    #                  'v_052206_120',
    #                  'v_052206_160',
    #                  'v_052206_200',
    #                  'v_052206_240',
    #                  'v_052206_280',
    #                  'v_052206_320',
    #                  'v_052206_360',
    #                  'v_052206_40',
    #                  'v_052206_80'],
    #  'train_dir_num': 'train_dataset_part1',
    #  'item_id': '052206'}
    filename_x1, filename_x2, train_dir_num, item_id = dataset_dict['filename_x1'], dataset_dict['filename_x2'], \
                                                       dataset_dict['train_dir_num'], dataset_dict['item_id']
    assert len(filename_x1) or len(filename_x2) or len(train_dir_num)
    global DATA_ROOT_DIR, IS_LOCAL
    a1, a2 = filename_x1.strip().split('_')[1], filename_x1.strip().split('_')[2]
    tmpfilename = os.path.join(os.path.abspath('.'), DATA_ROOT_DIR, train_dir_num, 'image', a1, a2)
    image_x1 = utils.read_image(tmpfilename, format="BGR")
    image_x1 = torch.as_tensor(image_x1.astype("float32").transpose(2, 0, 1))
    video_images = os.path.join(os.path.abspath('.'), DATA_ROOT_DIR, train_dir_num, 'video_images')
    if IS_LOCAL or os.path.exists(video_images):
        try:
            images_x2 = [utils.read_image(os.path.join(video_images, x) + '.jpg').astype("float32").transpose(2, 0, 1)
                         for x in filename_x2]
        except:
            images_x2 = []
    else:
        import cv2
        images_x2 = []
        video_path = os.path.join(DATA_ROOT_DIR, train_dir_num, 'video')
        video = cv2.VideoCapture(video_path)
        for frame_slice in filename_x2:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_slice.strip().split('_')[-1])
            _, frame_img = video.read()
            images_x2.append(frame_img.astype("float32").transpose(2, 0, 1))
        video.release()

    # image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    # dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    #
    # annos = [
    #     utils.transform_instance_annotations(obj, transforms, image.shape[:2])
    #     for obj in dataset_dict.pop("annotations")
    #     if obj.get("iscrowd", 0) == 0
    # ]
    # instances = utils.annotations_to_instances(annos, image.shape[:2])
    # dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return {
        'train_x1': image_x1,
        'train_x2': torch.as_tensor(images_x2),
        'item_id': item_id,
    }


def get_roi_feat(images, od_model):
    # with torch.no_grad:
    od_model.eval()
    image_tensors = ImageList.from_tensors([images], 32)
    features = od_model.backbone(image_tensors.tensor.cuda())
    proposals, _ = od_model.proposal_generator(image_tensors, features)
    instances = od_model.roi_heads._forward_box(features, proposals)
    features = [features[f] for f in od_model.roi_heads.in_features]
    mask_features = od_model.roi_heads.box_pooler(features, [x.pred_boxes for x in instances])
    return mask_features


def do_train(cfg, od_model, mmmodel, resume=False):
    mmmodel.train()
    od_model.eval()
    for d in [
        cfg.DATASETS.TRAIN[0],
        # cfg.DATASETS.TEST[0,
    ]:
        DatasetCatalog.register(d, lambda d=d: get_trainval_pos_pair_dicts(d))
        MetadataCatalog.get(d).set(thing_classes=["cloth"])
        if d.endswith('val'):
            MetadataCatalog.get(d).evaluator_type = "coco"

    # optimizer = optim.SGD(mmmodel.parameters(), lr=0.0025, momentum=0.9)
    optimizer = build_optimizer(cfg, mmmodel)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        mmmodel, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).
            get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    logger.info("Starting training from iteration {}".format(start_iter))
    criterion = nn.CrossEntropyLoss()

    with EventStorage(start_iter) as storage:
        for batch, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration += 1
            start_time = time.time()
            storage.step()
            label = torch.tensor([1.0], dtype=torch.long).cuda()
            pos_pair_num, running_loss = 0, 0.0
            for data in batch:
                try:
                    x1_fec = get_roi_feat(data['train_x1'].cuda(), od_model)[0].unsqueeze(0)
                    x1_fec = x1_fec.detach()
                except:
                    continue
                for x2s in data['train_x2']:
                    try:
                        x2s_fec = get_roi_feat(x2s.cuda(), od_model)[0].unsqueeze(0)
                        x2s_fec = x2s_fec.detach()
                    except:
                        continue
                    pos_pair_num += 1
                    # loss_dict = model(data)
                    # print(x1_fec.shape, '======================', x2s_fec.shape)
                    optimizer.zero_grad()
                    output = mmmodel(x1_fec.cuda(), x2s_fec.cuda())
                    loss = criterion(output, label)
                    assert torch.isfinite(loss).all()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
                chk_file_name = os.path.join(os.path.abspath('.'), cfg.OUTPUT_DIR, f'checkpoint_{iteration}.pth')
                logger.info(f'saved model to :{chk_file_name}')
                torch.save(mmmodel.state_dict(), chk_file_name)
            # if iteration - start_iter > 5 and (iteration % 100 == 0 or iteration == max_iter):
            #     torch.save(mmmodel, os.path.join(os.path.abspath('.'), cfg.OUTPUT_DIR) + f'checkpoint_{iteration}.pth')
            # 'mm_outputs/checkpoint_' + str(epoch + 1) + '.pth')
            periodic_checkpointer.step(iteration)
            end_time = time.time()
            logger.info(f'[iteration:{iteration}, Pos_pair_num:{pos_pair_num}, Neg_pair_num:{0}] \
                loss: {running_loss / pos_pair_num:.4f}, time(s): {end_time - start_time}')

            # {
            #     'filename_x1': image_x1,
            #     'filename_x1': torch.as_tensor(images_x2),
            #     'item_id': item_id,
            # }
            # x1 = data['train_x1']
            # x2 = data['train_x2']
            # print(x1.shape, x2.shape)
            # loss_dict = model(data)
            # losses = sum(loss_dict.values())
            # assert torch.isfinite(losses).all(), loss_dict
    return 0


def main(args):
    MetadataCatalog.get("tianchi_val").set(thing_classes=["cloth"])
    # # MetadataCatalog.get("tianchi_val").evaluator_type = "coco"
    # balloon_metadata = MetadataCatalog.get("tianchi_val")
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join('/opt/gitserial/tianchi/output', "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TEST = ("tianchi_val",)
    od_model = build_model(cfg).cuda()
    DetectionCheckpointer(od_model).load(cfg.MODEL.WEIGHTS)
    # logging.getLogger().disabled = True
    mmcfg = get_cfg()
    mmcfg.OUTPUT_DIR = 'mmoutput'
    mmcfg.DATASETS.TRAIN = ("match_train",)
    mmcfg.DATASETS.TEST = ("match_val",)
    mmcfg.DATALOADER.NUM_WORKERS = 2
    # 从 Model Zoo 中获取预训练模型
    mmcfg.MODEL.WEIGHTS = "mmoutput/checkpoint_1000.pth"
    mmcfg.SOLVER.IMS_PER_BATCH = 32
    mmcfg.MODEL.MASK_ON = False
    mmcfg.SOLVER.BASE_LR = 0.00025  # 学习率
    mmcfg.SOLVER.MAX_ITER = 3750  # 最大迭代次数 30000/8
    mmcfg.SOLVER.MAX_ITER = 1050  # 最大迭代次数 30000/32
    mmcfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    mmcfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别：红绿灯
    mmcfg.NUM_GPUS = 2
    mmcfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    os.makedirs(mmcfg.OUTPUT_DIR, exist_ok=True)
    mmmodel = MatchModel().cuda()
    mmmodel.load_state_dict(torch.load(mmcfg.MODEL.WEIGHTS))
    # distributed = comm.get_world_size() > 1
    # if distributed:
    #     od_model = DistributedDataParallel(od_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)
    #     mmmodel = DistributedDataParallel(mmmodel, device_ids=[comm.get_local_rank()], broadcast_buffers=False)
    logger.info(mmcfg)
    return do_train(mmcfg, od_model, mmmodel, resume=True)


if __name__ == "__main__":
    # ./train_match.py --num-gpu 2 --resume
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    logger.info(f"Command Line Args:{args}")
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


    ddd