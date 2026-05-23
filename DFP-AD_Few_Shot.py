import torch
import torch.nn as nn
import numpy as np
import os
import random
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive,distance_weighted_decay_loss, setup_seed, get_logger

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from utils import augmentation
# Model-Related Modules
from models import vit_encoder
from models.uad import DFP_AD
from models.vision_transformer import Mlp, Prototype_Learning_Block, Prototype_Aligned_Frequency_Modulator, bMlp
from thop import profile

warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)

            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            # Randomly select shot samples per class
            train_data.samples = [(sample[0], i) for sample in random.sample(train_data.samples, args.shot)]
            print_fn(train_data.samples)
            # Data augmentation
            augmented_data = []
            for img, label in train_data:
                augmented_imgs = augmentation(img)
                for aug_img in augmented_imgs:
                    augmented_data.append((aug_img, label))
            augmented_dataset = torch.utils.data.TensorDataset(
                torch.stack([item[0] for item in augmented_data]),
                torch.tensor([item[1] for item in augmented_data])
            )
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            train_data_list.append(augmented_dataset)
            test_data_list.append(test_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    elif args.dataset == 'Real-IAD' :
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                        gt_transform=gt_transform,
                                        phase='train')
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            # Randomly select shot samples per class
            train_data.img_paths = [sample for sample in random.sample(list(train_data.img_paths), args.shot)]
            print_fn(train_data.img_paths)
            # Data augmentation
            augmented_data = []
            for img, label in train_data:
                augmented_imgs = augmentation(img)
                for aug_img in augmented_imgs:
                    augmented_data.append((aug_img, label))
            augmented_dataset = torch.utils.data.TensorDataset(
                torch.stack([item[0] for item in augmented_data]),
                torch.tensor([item[1] for item in augmented_data])
            )
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform,
                                       phase="test")
            train_data_list.append(augmented_dataset)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                       drop_last=True)
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    PAFM_Decoder = []
    Prototype_Extractor = []

    # bottleneck
    Bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4))
    Bottleneck = nn.ModuleList(Bottleneck)

    # Initialize tokens
    Low_token = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))
                                for _ in range(1)])

    High_token = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))
                                 for _ in range(1)])

    # Prototype Learning
    for i in range(1):
        blk = Prototype_Learning_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        Prototype_Extractor.append(blk)
    Prototype_Extractor = nn.ModuleList(Prototype_Extractor)

    # PAFM_Decoder
    for i in range(8):
        blk = Prototype_Aligned_Frequency_Modulator(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        PAFM_Decoder.append(blk)
    PAFM_Decoder = nn.ModuleList(PAFM_Decoder)

    model = DFP_AD(
        encoder=encoder,
        bottleneck=Bottleneck,
        aggregation=Prototype_Extractor,
        decoder=PAFM_Decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token1=Low_token,
        prototype_token2=High_token
    )

    model = model.to(device)
    dummy_input = torch.randn(1, 3, 392, 392, dtype=torch.float).to(device)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, PAFM_Decoder, Prototype_Extractor, Low_token, High_token])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                           total_iters=args.total_epochs * len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        best_score = -1.0
        best_epoch = -1

        best_metrics = {
            "I-AUROC": (-1.0, -1),
            "I-AP": (-1.0, -1),
            "I-F1": (-1.0, -1),
            "P-AUROC": (-1.0, -1),
            "P-AP": (-1.0, -1),
            "P-F1": (-1.0, -1),
            "P-AUPRO": (-1.0, -1),
            "Mean": (-1.0, -1)
        }

        metric_names = list(best_metrics.keys())

        # 训练循环
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss= model(img)
                loss = distance_weighted_decay_loss(en, de) + g_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()

            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.total_epochs, np.mean(loss_list)))

            if (epoch + 1) % 5 == 0:

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(args.item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0
                    )
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                mean_results = [
                    np.mean(auroc_sp_list),
                    np.mean(ap_sp_list),
                    np.mean(f1_sp_list),
                    np.mean(auroc_px_list),
                    np.mean(ap_px_list),
                    np.mean(f1_px_list),
                    np.mean(aupro_px_list),
                ]
                avg_score = np.mean(mean_results)

                print_fn(
                    'Mean: I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        *mean_results))

                save_path = os.path.join(args.save_dir, args.save_name, f'model_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                print_fn(f"Model saved at epoch {epoch + 1} -> {save_path}")

                for name, val in zip(metric_names[:-1], mean_results):
                    if val > best_metrics[name][0]:
                        best_metrics[name] = (val, epoch + 1)

                if avg_score > best_metrics["Mean"][0]:
                    best_metrics["Mean"] = (avg_score, epoch + 1)

                if avg_score > best_score:
                    best_score = avg_score
                    best_epoch = epoch + 1
                    best_path = os.path.join(args.save_dir, args.save_name, 'model_best.pth')
                    torch.save(model.state_dict(), best_path)
                    print_fn(f"New best model saved at epoch {best_epoch} with avg_score {best_score:.4f}")

        print_fn("Best metrics and their epochs:")
        for name, (val, e) in best_metrics.items():
            print_fn(f"{name}: {val:.4f} at epoch {e}")

        print_fn(f"Best epoch (mean score): {best_epoch}, best average score: {best_score:.4f}")

    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        model.eval()

        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=0)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'E:\IMSN-LW\dataset\mvtec_anomaly_detection')  # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='DFP-AD-Few-Shot')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)  #4-shot:16  2-shot:8  1-shot: 4
    parser.add_argument('--shot', type=int, default=4) # Number of samples
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'-{args.shot}_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # category info
    if args.dataset == 'MVTec-AD':
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    elif args.dataset == 'VisA':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    elif args.dataset == 'Real-IAD':
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    main(args)
