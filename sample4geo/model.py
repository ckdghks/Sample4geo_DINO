import torch
import timm
import numpy as np
import torch.nn as nn


class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        def extract_cls_and_patch(x):
            features = self.model.forward_features(x)

            if features.dim() == 3:  # ViT 계열: (B, N+1, D)
                cls_token = features[:, 0]         # (B, D)
                patch_tokens = features[:, 1:]     # (B, N, D)
            elif features.dim() == 4:  # CNN 계열: (B, C, H, W)
                B, C, H, W = features.shape
                cls_token = features.mean(dim=[2, 3])  # GAP → (B, C)
                patch_tokens = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

            return cls_token, patch_tokens

        if img2 is not None:
            cls1, patch1 = extract_cls_and_patch(img1)
            cls2, patch2 = extract_cls_and_patch(img2)
            return cls1, cls2, patch1, patch2
        else:
            cls, patch = extract_cls_and_patch(img1)
            return cls, patch

class DINOModel(nn.Module):
    def __init__(self, backbone_name, img_size=384):
        super().__init__()
        self.student = TimmModel(backbone_name, pretrained=True, img_size=img_size)
        self.teacher = TimmModel(backbone_name, pretrained=True, img_size=img_size)
        
        # EMA를 위한 초기화
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self, momentum=0.996):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1. - momentum) * student_param.data

    def forward(self, x):
        return self.student(x)