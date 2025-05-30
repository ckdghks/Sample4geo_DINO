import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from sample4geo.trainer import train_with_distill
from sample4geo.utils import setup_system, Logger
from sample4geo.evaluate.university import evaluate
from sample4geo.loss import InfoNCE, dino_loss
from sample4geo.model import TimmModel

@dataclass
class Configuration:
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384
    mixed_precision: bool = True
    custom_sampling: bool = True
    seed = 1
    epochs: int = 40
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0,1)
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1
    clip_grad = 100.
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    label_smoothing: float = 0.1
    lr: float = 0.001
    scheduler: str = "cosine"
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001
    dataset: str = 'U1652-D2S'
    data_folder: str = "/workspace/mount/SSD_2T_a/AAM_Data/University_dataset/University-Release"
    prob_flip: float = 0.5
    model_path: str = "./university"
    zero_shot: bool = False
    checkpoint_start = None
    num_workers: int = 0 if os.name == 'nt' else 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    distill_start_epoch: int = 10
    distill_weight: float = 1.0
    teacher_momentum: float = 0.996


config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_train = '/workspace/mount/SSD_2T_a/AAM_Data/University_dataset/University-Release/train/drone' 
    config.gallery_folder_train = '/workspace/mount/SSD_2T_a/AAM_Data/University_dataset/University-Release/train/satellite'
    config.query_folder_test = '/workspace/mount/SSD_2T_a/AAM_Data/University_dataset/University-Release/test/query_drone' 
    config.gallery_folder_test = '/workspace/mount/SSD_2T_a/AAM_Data/University_dataset/University-Release/test/gallery_satellite'  
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = './data/U1652/train/satellite'
    config.gallery_folder_train = './data/U1652/train/drone'
    config.query_folder_test = './data/U1652/test/query_satellite'
    config.gallery_folder_test = './data/U1652/test/gallery_drone'


if __name__ == '__main__':

    model_path = f"{config.model_path}/{config.model}/{time.strftime('%H%M%S')}"
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.basename(__file__), f"{model_path}/train.py")
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    print("\nModel:", config.model)
    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    data_config = model.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model.load_state_dict(torch.load(config.checkpoint_start), strict=False)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    train_dataset = U1652DatasetTrain(config.query_folder_train, config.gallery_folder_train,
                                      train_sat_transforms, train_drone_transforms,
                                      config.prob_flip, config.batch_size)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    query_dataset_test = U1652DatasetEval(config.query_folder_test, mode="query", transforms=val_transforms)
    query_dataloader_test = DataLoader(query_dataset_test, batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers, shuffle=False, pin_memory=True)

    gallery_dataset_test = U1652DatasetEval(config.gallery_folder_test, mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n)

    gallery_dataloader_test = DataLoader(gallery_dataset_test, batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers, shuffle=False, pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn, device=config.device)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(len(train_dataloader) * config.warmup_epochs)

    if config.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, train_steps, config.lr_end, 1.5, warmup_steps)
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, train_steps, warmup_steps)
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    else:
        scheduler = None

    if config.zero_shot:
        evaluate(config, model, query_dataloader_test, gallery_dataloader_test, [1, 5, 10], step_size=1000, cleanup=True)

    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    teacher_model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    teacher_model.load_state_dict(model.module.state_dict())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids=config.gpu_ids)
    teacher_model = teacher_model.to(config.device)
    teacher_model.eval()

    best_score = 0
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'-'*30}[Epoch: {epoch}]{'-'*30}")

        train_loss = train_with_distill(config, model, teacher_model,
                                        train_dataloader, loss_function, dino_loss,
                                        optimizer, scheduler, scaler, epoch)

        print(f"Epoch: {epoch}, Train Loss = {train_loss:.3f}, Lr = {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            r1_test = evaluate(config, model, query_dataloader_test, gallery_dataloader_test, [1, 5, 10], step_size=1000, cleanup=True)
            if r1_test > best_score:
                best_score = r1_test
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    final_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(final_state, f'{model_path}/weights_end.pth')