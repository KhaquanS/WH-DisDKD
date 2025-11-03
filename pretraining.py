import torch
import torch.nn as nn


class Pretraining(nn.Module):
    """
    Standard pretraining of a teacher model.
    
    This method trains a teacher model using only the cross-entropy loss
    on ground truth labels. The model is fully unfrozen and can optionally
    start from pretrained weights.
    
    Args:
        teacher (nn.Module): Teacher model to be pretrained.
    """
    def __init__(self, teacher):
        super(Pretraining, self).__init__()
        self.teacher = teacher
        
        # Ensure all teacher parameters are trainable (unfrozen)
        for param in self.teacher.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass that computes only the teacher output.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            tuple: (None, teacher_logits, None)
                   First and third elements are None to maintain compatibility 
                   with other methods that return (teacher_logits, student_logits, method_loss).
        """
        # Forward pass through the teacher model
        teacher_logits = self.teacher(x)
        
        # Return None for student logits and method loss to maintain compatibility
        # Note: we return teacher_logits as the second element (student_logits position)
        # since that's what the training loop expects to compute loss against targets
        return None, teacher_logits, None