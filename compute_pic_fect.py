from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import os,torch,glob,pickle
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    detection_utils as utils,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
import cv2
import logging
import copy
from tqdm import tqdm as tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d  -   %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

root_dir='/opt/gitserial/tianchi'
with open(os.path.join(root_dir,'cache','train_image_annotation.pkl'),'rb') as fid:
    train_image_cache_dict=pickle.load(fid)

od_cfg = get_cfg()
od_cfg.merge_from_file(os.path.join(root_dir,"configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
od_cfg.DATALOADER.NUM_WORKERS = 2
od_cfg.MODEL.WEIGHTS = os.path.join(root_dir,'model','od_final.pth')
od_cfg.MODEL.MASK_ON = False
od_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
od_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
od_cfg.NUM_GPUS = 2
device=torch.device('cuda:1')
od_model = build_model(od_cfg).to(device)
DetectionCheckpointer(od_model).load(od_cfg.MODEL.WEIGHTS)
od_model.eval()
def get_roi_feat(images, od_model):
    od_model.eval()
    with torch.no_grad():
        image_tensors = ImageList.from_tensors([images], 32)
        features = od_model.backbone(image_tensors.tensor.to(device))
        proposals, _ = od_model.proposal_generator(image_tensors, features)
        instances = od_model.roi_heads._forward_box(features, proposals)
        features = [features[f] for f in od_model.roi_heads.in_features]
        mask_features = od_model.roi_heads.box_pooler(features, [x.pred_boxes for x in instances])
        return mask_features.detach()

result_fec_dict={}
for _,k in enumerate(tqdm(train_image_cache_dict.keys())):
    item_fec_list=[]
    for v in train_image_cache_dict[k]:
        file_name=v['file_name']
        for ann in v['annotation']['annotations']:
            viewpoint=ann['viewpoint']  #0：正面，1：背面，2：左侧，3：右侧）
            instance_id=ann['instance_id']
            if viewpoint > 1 or not instance_id :
                continue
            img = torch.as_tensor(utils.read_image(file_name,'BGR').astype("float32").transpose(2, 0, 1))
            item_fec_list.append(get_roi_feat(img,od_model).mean(axis=0))
    try:
        result_fec_dict[k] = torch.cat([x.unsqueeze(0) for x in item_fec_list]).mean(axis=0).unsqueeze(0).cpu()
    except:
        continue

out_cache_file='item_fect_dict.pkl'
with open(os.path.join(root_dir,'cache',out_cache_file),'wb') as fid:
    pickle.dump(result_fec_dict,fid,pickle.HIGHEST_PROTOCOL)