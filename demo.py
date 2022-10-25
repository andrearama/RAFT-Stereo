import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import logging 

DEVICE = 'cuda'
def compute_diff_desp_lidar(disp, gt_lidar):
    focal_length = 2840.562197
    baseline = 658.280549/2840.562197
    depth = focal_length * baseline / (disp + 1e-9)
    valid = (gt_lidar > 3 ).astype(np.float32)
    valid *= (gt_lidar < 200 ).astype(np.float32)
    # dd = valid*gt_lidar + (1-valid)*depth
    # plt.imshow(dd)
    # plt.show()

    mae = np.sum(np.abs(depth - gt_lidar)*valid) / np.sum(valid)    
    return mae

def load_image(imfile, stack_3 = False):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    if stack_3 : 
        img = cv2.resize(img, (1280, 704) ) #needs to be /32
        img = np.stack([img]*3, axis = -1)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    data_modality = "1 Passive Gated"
    model = torch.nn.DataParallel(RAFTStereo(args, data_modality), device_ids=[0])
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")        
        if data_modality != "All Gated" and False:
            checkpoint = torch.load(args.restore_ckpt)
            model.load_state_dict(checkpoint, strict=False)
            logging.info(f"Done loading checkpoint")
        else:
            current_model_dict = model.state_dict()
            loaded_state_dict = torch.load(args.restore_ckpt)
            new_state_dict={}
            for k,v in zip(current_model_dict.keys(), loaded_state_dict.values()) :
                if v.size()==current_model_dict[k].size() :
                    new_state_dict[k]  = v
                    print("----")
                else : 
                    print("a",v.shape)
                    print("b",current_model_dict[k].shape)
                    
                    new_state_dict[k] = current_model_dict[k]
            model.load_state_dict(new_state_dict, strict=False)            
            logging.info(f"Done loading checkpoint,some layers are not pretrained!!")
    asdads
    model = model.module
    model.to(DEVICE)
    model.eval()

    indexes_file = "/home/dense/Documents/andrea/GATED/test_gatedstereo.txt"
    with open(indexes_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]    
    
    # get input
    USE_GATED = True
    print("start getting imgs paths")
    img_names = []
    gt_names = []
    root_folder = "/external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3/"
    for l in lines : 
        day, ind = l.split(",")
        if not USE_GATED : #USE RGB
            full_path = sorted(glob.glob(root_folder + day + "/cam_stereo/left/image_rect/"+ind+"*.png"))
            full_path_right = sorted(glob.glob(root_folder + day + "/cam_stereo/right/image_rect/"+ind+"*.png"))
            gt_full_path = sorted(glob.glob(root_folder + day + "/cam_stereo/left/lidar_vls128_projected/"+ind+"*.npz"))
        else:
            type_gated = "type7"
            full_path = sorted(glob.glob(root_folder + day + "/framegrabber/left/bwv/"+type_gated+"/image_rect8/"+ind+"*.png"))
            full_path_right = sorted(glob.glob(root_folder + day + "/framegrabber/right/bwv/"+type_gated+"/image_rect8/"+ind+"*.png"))
            gt_full_path = sorted(glob.glob(root_folder + day + "/framegrabber/left/lidar_vls128_projected/"+ind+"*.npz"))

        
        print(root_folder + day + "/cam_stereo/left/image_rect/"+ind+"*.png")
        if len(full_path) == 1 and (len(gt_full_path) == 1) and len(full_path_right) == 1:
            full_path = full_path[0]
            img_names.append([full_path,full_path_right[0],day])
            gt_names.append(gt_full_path[0])
        else:
            pass#error


    with torch.no_grad():
        MAE_v = []
        for img_name_day, gt_lidar in tqdm(zip(img_names, gt_names)):
            img_name, img_name_right, day = img_name_day

            if os.path.isdir(img_name):
                continue

            print("processing",img_name )
            print("gt",gt_lidar )
            image1 = load_image(img_name, stack_3=USE_GATED)
            image2 = load_image(img_name_right, stack_3=USE_GATED)

            depth_gt = np.load(gt_lidar)["arr_0"]
            if USE_GATED :
                depth_gt = cv2.resize(depth_gt, (1280, 704) ) #needs to be /32

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            final_disp = np.abs(flow_up.cpu().numpy().squeeze())
            MAE = compute_diff_desp_lidar(final_disp, depth_gt)
            MAE_v.append(MAE)
            print("MAE",MAE)

            plt.imshow(final_disp)
            plt.show()             
            asdasd
            #continue

            model_type = args.restore_ckpt.split("/")[-1].replace(".pth","")
            output_path = "/external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3"

            path_path = os.path.join(output_path,day,"cam_stereo","left", model_type)
            if not os.path.isdir( path_path ) : 
                os.mkdir( path_path )
                
            if not os.path.isdir(os.path.join(path_path,"visualization") ) : 
                os.mkdir(os.path.join(path_path,"visualization") )

            if not os.path.isdir(os.path.join(path_path,"npy") ) : 
                os.mkdir(os.path.join(path_path,"npy") )
                
            filename = os.path.join(
                path_path,"visualization", os.path.splitext(os.path.basename(img_name))[0]
            )+".png"

            print("saving:",filename)

            focal_length = 2840.562197
            baseline = 658.280549/2840.562197
            depth = focal_length * baseline / (final_disp + 1e-9)

            # plt.imshow(depth)
            # plt.show()           

            np.save(filename.replace("/visualization/","/npy/").replace(".png",".npy"), depth)
            plt.imsave(filename, depth , cmap='jet')
            
    print("AVG MAGE:",sum(MAE_v)/len(MAE_v) )        

if __name__ == '__main__':

    # left = '/external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3/2022-10-12_15-25-15/cam_stereo/left/image_rect/01449_1665588464987378730.png'
    # right = left.replace("/left/","/right/")


    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None )
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None )
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
