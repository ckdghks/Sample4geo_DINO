import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 
def dino_loss(student_feat, teacher_feat, temp_s=0.1, temp_t=0.07):
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)

    # stop gradient on teacher
    teacher_feat = teacher_feat.detach()

    student_logits = student_feat / temp_s
    teacher_logits = teacher_feat / temp_t

    return F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')


class PatchSimLoss(nn.Module):
    """
    Patch-level Similarity Structure Loss
    구조: teacher와 student의 patch간 cosine similarity 구조를 정렬함.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_patch_feat, teacher_patch_feat):
        """
        student_patch_feat: (B, N, D)  -> N개 patch의 feature
        teacher_patch_feat: (B, N, D)
        """
        B, N, D = student_patch_feat.shape

        # Normalize
        student_patch_feat = F.normalize(student_patch_feat, dim=-1)  # (B, N, D)
        teacher_patch_feat = F.normalize(teacher_patch_feat, dim=-1)

        # Similarity Matrix 계산: (B, N, N)
        student_sim = torch.matmul(student_patch_feat, student_patch_feat.transpose(1, 2)) / self.temperature
        teacher_sim = torch.matmul(teacher_patch_feat, teacher_patch_feat.transpose(1, 2)) / self.temperature

        # Softmax로 정규화
        student_sim = F.log_softmax(student_sim, dim=-1)
        teacher_sim = F.softmax(teacher_sim.detach(), dim=-1)

        # KL Divergence Loss
        loss = F.kl_div(student_sim, teacher_sim, reduction='batchmean')
        return loss

class GlobalStructureSimLoss(nn.Module):
    """
    Patch-to-patch 전체 구조 유사도 행렬 정합 (ConvNeXt, CNN-friendly)
    입력: (B, N, D) 형태의 flatten된 patch-level feature
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_patch_feat, teacher_patch_feat):
        """
        student_patch_feat, teacher_patch_feat: (B, N, D)
        """
        # Normalize features
        student_patch_feat = F.normalize(student_patch_feat, dim=-1)
        teacher_patch_feat = F.normalize(teacher_patch_feat, dim=-1)

        # Similarity matrix: (B, N, N)
        sim_student = torch.matmul(student_patch_feat, student_patch_feat.transpose(1, 2))
        sim_teacher = torch.matmul(teacher_patch_feat, teacher_patch_feat.transpose(1, 2))

        # MSE loss over similarity matrices
        loss = F.mse_loss(sim_student, sim_teacher)
        return loss