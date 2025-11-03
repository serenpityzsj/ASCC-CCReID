"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ICASSP 2025 paper: Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification
URL: arxiv.org/abs/2405.16597
GitHub: https://github.com/QizaoWang/CSSC-CCReID
"""

from tqdm import tqdm
from utils.util import AverageMeter

def log_alpha_multi(model, epoch, log_path='logs/alpha_log_cele.txt'):
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        a1 = model.smr_c_branch1.wavelet_division.alpha.item()
        a2 = model.smr_s_branch1.wavelet_division.alpha.item()
        a3 = model.smr_s_branch2.wavelet_division.alpha.item()
        a4 = model.smr_c_branch2.wavelet_division.alpha.item()

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch:03d}: ¦Á_C1={a1:.4f}, ¦Á_S1={a2:.4f}, ¦Á_S2={a3:.4f}, ¦Á_C2={a4:.4f}\n")

        print(f"[Epoch {epoch}] ¦Á_C1={a1:.4f} | ¦Á_S1={a2:.4f} | ¦Á_S2={a3:.4f} | ¦Á_C2={a4:.4f}")
    except Exception as e:
        print(f"[Warning] Alpha logging failed: {e}")


def train(args, epoch, train_loader, model, optimizer, scheduler, class_criterion, metric_criterion, use_gpu):
    tri_start_epoch = args.tri_start_epoch
    id_losses = AverageMeter()
    tri_losses = AverageMeter()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if args.dataset in ['prcc', 'ltcc']:
            img, pid, _, _ = data
        else:
            img, pid, _ = data

        if use_gpu:
            img, pid = img.cuda(), pid.cuda()

        model.train()
        feat_list, y_list = model(img)

        loss = 0
        for y in y_list:
            id_loss = class_criterion(y, pid)
            loss += id_loss

        if epoch > tri_start_epoch:
            for feat in feat_list:
                tri_loss = metric_criterion(feat, pid)
                loss += tri_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        id_losses.update(id_loss.item(), pid.size(0))
        if epoch > tri_start_epoch:
            tri_losses.update(tri_loss.item(), pid.size(0))

    if args.print_train_info_epoch_freq != -1 and epoch % args.print_train_info_epoch_freq == 0:
        print('Ep{0} Id:{id_loss.avg:.4f} Tri:{tri_loss.avg:.4f} '.format(
            epoch, id_loss=id_losses, tri_loss=tri_losses))
        # try:
        #     log_alpha_multi(model, epoch)
        # except Exception as e:
        #     print(f"[Alpha Log Warning] {e}")

    scheduler.step()


