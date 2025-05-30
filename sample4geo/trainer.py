import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

def train_with_distill(train_config, model, teacher_model, dataloader, loss_function, dino_loss_fn, optimizer, scheduler=None, scaler=None, epoch=0):
    model.train()
    teacher_model.eval()

    losses = AverageMeter()
    time.sleep(0.1)
    optimizer.zero_grad(set_to_none=True)

    step = 1
    bar = tqdm(dataloader, total=len(dataloader)) if train_config.verbose else dataloader

    for query, reference, ids in bar:
        query = query.to(train_config.device)
        reference = reference.to(train_config.device)

        if scaler:
            with autocast():
                features1, features2 = model(query, reference)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss_main = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss_main = loss_function(features1, features2, model.logit_scale.exp())

                if epoch >= train_config.distill_start_epoch:
                    with torch.no_grad():
                        t_feat1, t_feat2 = teacher_model(query, reference)
                    loss_distill = (dino_loss_fn(features1, t_feat1) + dino_loss_fn(features2, t_feat2)) / 2
                    loss = loss_main + train_config.distill_weight * loss_distill
                else:
                    loss = loss_main
                losses.update(loss.item())

            scaler.scale(loss).backward()
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss_main = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss_main = loss_function(features1, features2, model.logit_scale.exp())

            if train_config.epoch >= train_config.distill_start_epoch:
                with torch.no_grad():
                    t_feat1, t_feat2 = teacher_model(query, reference)
                loss_distill = (dino_loss_fn(features1, t_feat1) + dino_loss_fn(features2, t_feat2)) / 2
                loss = loss_main + train_config.distill_weight * loss_distill
            else:
                loss = loss_main
            losses.update(loss.item())

            loss.backward()
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        if train_config.scheduler in ["polynomial", "cosine", "constant"]:
            scheduler.step()

        if train_config.verbose:
            bar.set_postfix(ordered_dict={"loss": f"{loss.item():.4f}", "loss_avg": f"{losses.avg:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg

def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list