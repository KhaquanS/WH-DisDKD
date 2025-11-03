import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    
    Args:
        named_layers (list of tuples): List of tuples in the form (layer_name, layer_module)
    """
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []
        
        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output
            return _hook
        
        # Register a forward hook for each named layer.
        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    def clear(self):
        """Clears the stored features."""
        self.features.clear()
        
    def remove(self):
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class HintLoss(nn.Module):
    """
    Computes an MSE loss between the teacher's features and the student's features,
    with optional adaptation of the student's features to match the teacher's dimensions.
    
    Args:
        teacher_channels (int): Number of channels in the teacher's feature map.
        student_channels (int): Number of channels in the student's feature map.
        adapter (str): Specifies whether to attach the adapter on the student or the teacher.
    """
    def __init__(self, teacher_channels, student_channels, adapter):
        super(HintLoss, self).__init__()

        self.adapter = adapter
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels

        if teacher_channels == student_channels:
            self.adaptation = nn.Identity()
            return
        
        if adapter == 'student':
            self.adaptation = nn.Conv2d(
                student_channels, 
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ) 
        elif adapter == 'teacher':
            self.adaptation = nn.Conv2d(
                teacher_channels,
                student_channels, 
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ) 
        else:
            raise ValueError(f'{adapter} is not valid. Choose from student or teacher.')

        print(f'{adapter} adapter has {count_params(self.adaptation)} params...\n')

    def forward(self, teacher_features, student_features):
        # Adapt student's features.
        if self.adapter == 'student':
            adapted_student = self.adaptation(student_features)
            if teacher_features.shape != adapted_student.shape:
                adapted_student = F.interpolate(
                    adapted_student,
                    size=teacher_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return F.mse_loss(adapted_student, teacher_features)
        
        # Adapt teacher's features.
        elif self.adapter == 'teacher':
            adapted_teacher = self.adaptation(teacher_features)
            if student_features.shape != adapted_teacher.shape:
                adapted_teacher = F.interpolate(
                    adapted_teacher,
                    size=student_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return F.mse_loss(adapted_teacher, student_features)


class FitNet(nn.Module):
    """
    FitNet for knowledge distillation. This module trains a student network using
    hints from intermediate feature maps of a teacher network.
    
    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network to be trained.
        teacher_layer (str): Teacher layer name for feature extraction.
        student_layer (str): Student layer name for feature extraction.
        teacher_channels (int): Number of channels in teacher feature map.
        student_channels (int): Number of channels in student feature map.
        adapter (str): Type of adapter ('student' or 'teacher').
    """
    def __init__(self, teacher, student, teacher_layer, student_layer, 
                 teacher_channels, student_channels, adapter='student'):
        super(FitNet, self).__init__()
        self.teacher = teacher
        self.student = student
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        
        # Register hooks to capture intermediate features.
        self.teacher_hooks = FeatureHooks([
            (teacher_layer, get_module(self.teacher.model, teacher_layer))
        ])
        self.student_hooks = FeatureHooks([
            (student_layer, get_module(self.student.model, student_layer))
        ])
        
        # Create hint loss criterion
        self.hint_criterion = HintLoss(teacher_channels, student_channels, adapter)

    def forward(self, x):
        """
        Forward pass that computes both teacher and student outputs and hint loss.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            tuple: (teacher_logits, student_logits, hint_loss)
        """
        # Forward pass through teacher and student networks.
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Extract intermediate features
        teacher_feature = self.teacher_hooks.features.get(self.teacher_layer)
        student_feature = self.student_hooks.features.get(self.student_layer)

        if teacher_feature is None or student_feature is None:
            raise ValueError(f"Missing features for layers: {self.teacher_layer} or {self.student_layer}")
        
        # Compute hint loss
        hint_loss = self.hint_criterion(teacher_feature, student_feature)
        
        # Clear features for next forward pass
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return teacher_logits, student_logits, hint_loss