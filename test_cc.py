"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID

python main.py --gpu_devices 0 --dataset ltcc --dataset_root dataset/data --dataset_filename LTCC-reID --save_dir ltcc_save_21 --save_checkpoint
python main.py --gpu_devices 0 --dataset ltcc --dataset_root dataset/data --dataset_filename LTCC-reID  --resume ltcc_save/best_rank1_model.pth --save_dir ltcc_evaluate_1 --evaluate
"""
from thop import profile
from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from evaluate.metrics import evaluate
from test import get_distmat
from evaluate.metrics_for_cc import evaluate_ltcc, evaluate_prcc_all_gallery
from model.re_ranking import fast_gcrv_image

from easydict import EasyDict as edict

cfg = edict()
cfg.GCR = edict()

# Ìí¼ÓËùÐè²ÎÊý£¨±ØÐë°üº¬ÄãÓÃµ½µÄ£©
cfg.GCR.ENABLE_GCR = True
cfg.GCR.MODE = 'sym'            # or 'fixA'
cfg.GCR.GAL_ROUND = 1
cfg.GCR.LAMBDA1 = 4.2
cfg.GCR.LAMBDA2 = 4.2           # Äã¿ÉÄÜÒ²ÐèÒª¼ÓÕâ¸ö
cfg.GCR.BETA1 = 0.3
cfg.GCR.BETA2 = 0.3             # ?? Õâ¸ö¾ÍÊÇÏÖÔÚÈ±µÄ
cfg.GCR.SCALE = 1.0
cfg.GCR.WITH_GPU = True

cfg.COMMON = edict()
cfg.COMMON.VERBOSE = True

import numpy as np

def get_top10_rank1_pid(distmat, query_ids, gallery_ids, query_cams, gallery_cams):
    """
    Í³¼ÆÔÚ Rank-1 Æ¥ÅäÖÐÃüÖÐ×î¶àµÄ10¸ö query ID£¨PID£©

    ²ÎÊý:
        distmat: [num_query, num_gallery]£¬¾àÀë¾ØÕó£¬Ô½Ð¡Ô½ÏàËÆ
        query_ids: [num_query]£¬query ¶ÔÓ¦µÄ PID
        gallery_ids: [num_gallery]£¬gallery ¶ÔÓ¦µÄ PID
        query_cams: [num_query]£¬query ¶ÔÓ¦ÉãÏñÍ· ID
        gallery_cams: [num_gallery]£¬gallery ¶ÔÓ¦ÉãÏñÍ· ID

    ·µ»Ø:
        top10_pid: ÃüÖÐ´ÎÊý×î¶àµÄÇ°10¸ö PID
    """
    indices = np.argsort(distmat, axis=1)  # ÅÅÐò£¬Ã¿ÐÐ¶ÔÓ¦Ò»¸ö query
    rank1_hit_count = {}

    for i, q_pid in enumerate(query_ids):
        q_cam = query_cams[i]
        ranked_idx = indices[i]
        for g_idx in ranked_idx:
            g_pid = gallery_ids[g_idx]
            g_cam = gallery_cams[g_idx]
            if q_pid == g_pid and q_cam != g_cam:
                rank1_hit_count[q_pid] = rank1_hit_count.get(q_pid, 0) + 1
                break  # Ö»¼ÇÂ¼ rank-1 ÃüÖÐ
            # Èç¹ûÔÊÐí top-k Æ¥Åä¿ÉÒÔ×¢ÊÍ break

    # ÅÅÐò£¬È¡ top-10
    top10 = sorted(rank1_hit_count.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_pid = [pid for pid, _ in top10]

    print("[Top10 Rank-1 ÃüÖÐ×î¶àµÄ PID]", top10_pid)
    return top10_pid


def get_data_for_cc(datasetloader, use_gpu, model):
    with torch.no_grad():
        feats, pids, clothids, camids= [], [], [], []
        for batch_idx, (img, pid, clothid, camid) in enumerate(tqdm(datasetloader)):
            flip_img = torch.flip(img, [3])
            if use_gpu:
                img, flip_img = img.cuda(), flip_img.cuda()
            feat = model(img)
            feat_flip = model(flip_img)
            feat += feat_flip
            feat = F.normalize(feat, p=2, dim=1)
            feat = feat.data.cpu()
            feats.append(feat)
            pids.extend(pid)
            clothids.extend(clothid)
            camids.extend(camid)
        feats = torch.cat(feats, 0)
        pids = np.asarray(pids)
        clothids = np.asarray(clothids)
        camids = np.asarray(camids)
    return feats, pids, clothids, camids

#
# def test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model,
#                   use_gpu, ranks=[1, 5, 10], epoch=None):
#     model.eval()
#     gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
#     qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_sc_loader, use_gpu, model)
#     # distmat = get_distmat(qf, gf)
#     all_data = [qf, q_pids, q_camids, gf, g_pids, g_camids]
#     distmat, prb_feats_new, gal_feats_new = fast_gcrv_image(cfg, all_data)
#
#     cmc, mAP = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
#     if epoch: print("Epoch {}: ".format(epoch), end='')
#     print("mAP: {:.4%}  ".format(mAP), end='')
#     for r in ranks:
#         print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
#     print()
#
#     gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
#     qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_cc_loader, use_gpu, model)
#     # distmat = get_distmat(qf, gf)
#     all_data = [qf, q_pids, q_camids, gf, g_pids, g_camids]
#     distmat, prb_feats_new, gal_feats_new = fast_gcrv_image(cfg, all_data)
#     cmc_2, mAP_2 = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
#     if epoch: print("Epoch {}: ".format(epoch), end='')
#     print("mAP: {:.4%}  ".format(mAP_2), end='')
#     for r in ranks:
#         print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
#     # print()
#     #
#     # # print("With Reranking: ", end='')
#     # # print('--------------------------------------------------------------------------------------------------------------')
#     # #
#     # # all_data = [qf, q_pids, q_camids, gf, g_pids, g_camids]
#     # # distmat, prb_feats_new, gal_feats_new = fast_gcrv_image(cfg, all_data)
#     # # cmc, mAP = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
#     # # if epoch: print("Epoch {}: ".format(epoch), end='')
#     # # print("mAP: {:.4%}  ".format(mAP), end='')
#     # # for r in ranks:
#     # #     print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
#     # # print()
#     # #
#     # # qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_cc_loader, use_gpu, model)
#     # # all_data = [qf, q_pids, q_camids, gf, g_pids, g_camids]
#     # # distmat, prb_feats_new, gal_feats_new = fast_gcrv_image(cfg, all_data)
#     # # cmc_2, mAP_2 = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
#     # # if epoch: print("Epoch {}: ".format(epoch), end='')
#     # # print("mAP: {:.4%}  ".format(mAP_2), end='')
#     # # for r in ranks:
#     # #     print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
#     # # print()
#     #
#     # # print("With Reranking: ", end='')
#     # # print('--------------------------------------------------------------------------------------------------------------')
#     #
#     # return [cmc[0], cmc_2[0]], [mAP, mAP_2]

def test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model,
                  use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_sc_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)
    cmc, mAP = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_cc_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)
    cmc_2, mAP_2 = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP_2), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    print()

    return [cmc[0], cmc_2[0]], [mAP, mAP_2]

def test_for_ltcc(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    # dummy_input = torch.randn(1, 3, 384, 192).cuda()
    # # ÌáÈ¡Ô­Ê¼Ä£ÐÍÓÃÓÚ FLOPs ¼ÆËã
    # if hasattr(model, 'module'):
    #     model_for_profile = model.module
    # else:
    #     model_for_profile = model
    #
    # flops, params = profile(model_for_profile, inputs=(dummy_input,))
    # print(f"FLOPs: {flops / 1e9:.5f} G, Params: {params / 1e6:.5f} M")

    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_loader, use_gpu, model)
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)

    cmc, mAP = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
                             ltcc_cc_setting=False)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    cmc_2, mAP_2 = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
                             ltcc_cc_setting=True)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP_2), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    print()

    # print("With Reranking: ", end='')
    # print('--------------------------------------------------------------------------------------------------------------')
    #
    # all_data = [qf, q_pids, q_camids, gf, g_pids, g_camids]
    # distmat, prb_feats_new, gal_feats_new = fast_gcrv_image(cfg, all_data)
    # cmc, mAP = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
    #                          ltcc_cc_setting=False)
    # if epoch: print("Epoch {}: ".format(epoch), end='')
    # print("mAP: {:.4%}  ".format(mAP), end='')
    # for r in ranks:
    #     print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    # print()
    #
    # cmc_2, mAP_2 = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
    #                              ltcc_cc_setting=True)
    # if epoch: print("Epoch {}: ".format(epoch), end='')
    # print("mAP: {:.4%}  ".format(mAP_2), end='')
    # for r in ranks:
    #     print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    # print()
    #
    # top10_pid = get_top10_rank1_pid(
    #     distmat=distmat,
    #     query_ids=q_pids,
    #     gallery_ids=g_pids,
    #     query_cams=q_camids,
    #     gallery_cams=g_camids
    # )
    # print(top10_pid)

    return [cmc[0], cmc_2[0]], [mAP, mAP_2]


def test_for_cc(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_loader, use_gpu, model)
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    return cmc[0], mAP
