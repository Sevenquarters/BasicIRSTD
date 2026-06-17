# filepath: d:\Program Files (x86)\IRSTD\BasicIRSTD\test_gca.py
"""
Test script for GCA models - compatible with 'model.' prefix checkpoints.
Wrapper around test.py that handles checkpoint format conversion.
"""

import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test for GCA models")
parser.add_argument("--model_names", default=['ResUNet-GCA'], nargs='+',  
                    help="model_name: 'ResUNet-GCA', 'ResUNet-GCA-NoGate', etc.")
parser.add_argument("--pth_dirs", default=None, nargs='+',  
                    help="checkpoint dir")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], nargs='+', 
                    help="dataset_name")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg")
parser.add_argument("--save_img", default=True, type=bool, help="save image or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std


def load_state_dict_compatible(model, checkpoint_path, device='cuda'):
    """Load checkpoint, handling all variants (standard, NoGate ablation)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    # Debug info
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    print(f"  Model keys: {len(model_keys)}, Checkpoint keys: {len(state_dict)}")
    if missing:
        print(f"    Missing from checkpoint: {len(missing)} keys (e.g., {list(missing)[:2]})")
    if unexpected:
        print(f"    Unexpected in checkpoint: {len(unexpected)} keys (e.g., {list(unexpected)[:2]})")
    
    try:
        # Try strict loading first (works for most cases)
        model.load_state_dict(state_dict)
        print(f"  ✓ Loaded checkpoint: {checkpoint_path}")
    except RuntimeError as e:
        error_msg = str(e)
        
        # If only gate parameters are missing (NoGate ablation), use strict=False
        if "Missing key" in error_msg and "gate" in error_msg:
            print(f"  ⚠ Retrying with strict=False (missing gate keys)...")
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Loaded checkpoint: {checkpoint_path} (ablation mode)")
        
        # If unexpected keys, also try strict=False
        elif "Unexpected key" in error_msg:
            print(f"  ⚠ Retrying with strict=False (unexpected keys)...")
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Loaded checkpoint: {checkpoint_path} (loose mode)")
        
        else:
            print(f"\n  ✗ ERROR loading {checkpoint_path}:")
            print(f"  {error_msg[:300]}\n")
            raise


def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_state_dict_compatible(net, opt.pth_dir, device)
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
            
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')  
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')


if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_gca_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_best.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir + '\n')
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
