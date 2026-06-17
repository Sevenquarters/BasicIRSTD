import argparse
import csv
import glob
import re
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['ACM', 'ALCNet'], nargs='+', 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet', 'ResUNet', 'ResUNet-CBAM', 'ResUNet-GCA', 'ResUNet-GCA-DSPG', 'ResUNet-GCA-NoGate'")              
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, nargs='+', help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--intervals", type=int, default=10, help="Intervals for print loss")
parser.add_argument("--eval_intervals", type=int, default=1, help="Intervals for evaluation metrics")
parser.add_argument("--save_intervals", type=int, default=50, help="Intervals for checkpoint saving")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)

def _safe_float(text):
    try:
        return float(text)
    except Exception:
        return None

def load_history_from_log(log_path):
    history = []
    if not os.path.exists(log_path):
        return history

    epoch = None
    loss = None
    pixacc = None
    miou = None
    pd = None
    fa = None

    loss_re = re.compile(r"Epoch---(\d+), total_loss---([0-9eE\.\+\-]+)")
    miou_re = re.compile(r"pixAcc, mIoU:\s*\(\s*([0-9eE\.\+\-]+)\s*,\s*(?:np\.float64\()?\s*([0-9eE\.\+\-]+)\s*\)?\s*\)")
    pd_fa_re = re.compile(r"PD, FA:\s*\(\s*([0-9eE\.\+\-]+)\s*,\s*([0-9eE\.\+\-]+)\s*\)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = loss_re.search(line)
            if m:
                epoch = int(m.group(1))
                loss = _safe_float(m.group(2))
                continue

            m = miou_re.search(line)
            if m:
                pixacc = _safe_float(m.group(1))
                miou = _safe_float(m.group(2))
                continue

            m = pd_fa_re.search(line)
            if m and epoch is not None and loss is not None:
                pd = _safe_float(m.group(1))
                fa = _safe_float(m.group(2))
                history.append({
                    "epoch": epoch,
                    "loss": loss,
                    "pixAcc": pixacc,
                    "mIoU": miou,
                    "PD": pd,
                    "FA": fa,
                })
                epoch = None
                loss = None
                pixacc = None
                miou = None
                pd = None
                fa = None
    return history

def load_history_from_csv(csv_path):
    history = []
    if not os.path.exists(csv_path):
        return history

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                "epoch": int(row["epoch"]),
                "loss": _safe_float(row["loss"]),
                "pixAcc": _safe_float(row["pixAcc"]),
                "mIoU": _safe_float(row["mIoU"]),
                "PD": _safe_float(row["PD"]),
                "FA": _safe_float(row["FA"]),
            })
    return history

def write_history_csv(csv_path, history):
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "pixAcc", "mIoU", "PD", "FA"])
        writer.writeheader()
        for item in history:
            writer.writerow(item)

def append_history_csv(csv_path, item, header_if_missing=True):
    exists = os.path.exists(csv_path)
    if not exists and header_if_missing:
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "pixAcc", "mIoU", "PD", "FA"])
        if not exists and header_if_missing:
            writer.writeheader()
        writer.writerow(item)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    net = Net(model_name=opt.model_name, mode='train').to(device)
    net.train()
    
    epoch_state = 0
    total_loss_list = []
    metrics_history = []
    best_miou = -1.0
    best_epoch = 0
    metrics_csv = os.path.join(opt.save, f"{opt.dataset_name}_{opt.model_name}_metrics.csv")
    current_log_path = opt.f.name

    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                best_miou = ckpt.get('best_mIoU', best_miou)
                best_epoch = ckpt.get('best_epoch', best_epoch)
                if 'history' in ckpt and ckpt['history'] is not None:
                    metrics_history = ckpt['history']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if opt.dataset_name in pretrained_pth and opt.model_name in pretrained_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
    
    ### Default settings                
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}
        opt.scheduler_settings['epochs'] = opt.nEpochs
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}
        opt.scheduler_settings['epochs'] = opt.nEpochs
        
    opt.nEpochs = opt.scheduler_settings['epochs']

    net = torch.nn.DataParallel(net)
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)

    if len(metrics_history) == 0:
        metrics_history = load_history_from_csv(metrics_csv)
    if len(metrics_history) == 0 and opt.resume:
        previous_log = None
        log_pattern = os.path.join(opt.save, f"{opt.dataset_name}_{opt.model_name}_*.txt")
        candidates = [p for p in glob.glob(log_pattern) if os.path.abspath(p) != os.path.abspath(current_log_path)]
        if len(candidates) > 0:
            previous_log = max(candidates, key=os.path.getmtime)
        if previous_log is not None:
            metrics_history = load_history_from_log(previous_log)
    if len(metrics_history) > 0:
        write_history_csv(metrics_csv, metrics_history)
        if best_miou < 0:
            best_item = max(metrics_history, key=lambda x: (x["mIoU"] if x["mIoU"] is not None else -1))
            best_miou = best_item["mIoU"]
            best_epoch = best_item["epoch"]
        print(f"Loaded history: {len(metrics_history)} epochs, best mIoU={best_miou:.6f} at epoch {best_epoch}")

    for idx_epoch in range(epoch_state, opt.nEpochs):
        epoch_loss_values = []
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).to(device), Variable(gt_mask).to(device)
            if img.shape[0] == 1:
                continue
            pred = net.forward(img)
            loss = net.module.loss(pred, gt_mask)
            epoch_loss_values.append(float(loss.detach().cpu()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_loss = float(np.array(epoch_loss_values).mean()) if len(epoch_loss_values) > 0 else 0.0
        total_loss_list.append(epoch_loss)

        if (idx_epoch + 1) % opt.intervals == 0:
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
                  % (idx_epoch + 1, total_loss_list[-1]))

        if (idx_epoch + 1) % opt.eval_intervals == 0:
            print('Epoch---%d evaluation:' % (idx_epoch + 1))
            opt.f.write('Epoch---%d evaluation:\n' % (idx_epoch + 1))
            results1, results2 = test(state_dict=net.module.state_dict())
            eval_row = {
                "epoch": idx_epoch + 1,
                "loss": epoch_loss,
                "pixAcc": float(results1[0]),
                "mIoU": float(results1[1]),
                "PD": float(results2[0]),
                "FA": float(results2[1]),
            }
            metrics_history.append(eval_row)
            append_history_csv(metrics_csv, eval_row)

            current_miou = eval_row["mIoU"]
            if current_miou > best_miou:
                best_miou = current_miou
                best_epoch = idx_epoch + 1
                best_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_best.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                    'total_loss': total_loss_list,
                    'history': metrics_history,
                    'best_mIoU': best_miou,
                    'best_epoch': best_epoch,
                }, best_pth)
                print(f"Best model updated: epoch {best_epoch}, mIoU={best_miou:.6f}")

        if (idx_epoch + 1) % opt.save_intervals == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
                'total_loss': total_loss_list,
                'history': metrics_history,
                'best_mIoU': best_miou,
                'best_epoch': best_epoch,
                }, save_pth)
            
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % opt.save_intervals != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
                'total_loss': total_loss_list,
                'history': metrics_history,
                'best_mIoU': best_miou,
                'best_epoch': best_epoch,
                }, save_pth)

def test(save_pth=None, state_dict=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').to(device)
    if state_dict is None:
        ckpt = torch.load(save_pth)
        state_dict = ckpt['state_dict']
    net.load_state_dict(state_dict)
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            img = Variable(img).to(device)
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    return results1, results2
    
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
