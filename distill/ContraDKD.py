import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.utils import get_module, count_params

# --- Helper Classes (Copied for self-containment) ---


class FeatureHooks:
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []

        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output

            return _hook

        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FeatureRegressor(nn.Module):
    """1x1 Conv to project features to common dimension."""

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """Discriminator with Global Pooling."""

    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        return self.discriminator(pooled)


class L2CContrastiveLoss(nn.Module):
    """
    Conditional Contrastive Loss (L2C).
    Encourages features to cluster around learnable class proxies.
    """

    def __init__(self, num_classes, feat_dim=256, temperature=0.07):
        super(L2CContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        # Learnable Class Proxies
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.class_embeddings)

    def forward(self, features, labels):
        # features: [B, Dim] (Normalized)
        # labels: [B]

        # Normalize proxies
        proxies = F.normalize(self.class_embeddings, dim=1)

        # 1. Positive Proxy Similarity: (x_i . c_yi)
        # Gather the specific proxy for each label in the batch
        target_proxies = proxies[labels]
        proxy_sim = (features * target_proxies).sum(
            dim=1, keepdim=True
        ) / self.temperature

        # 2. Batch Similarity Matrix: (x_i . x_j)
        batch_sim = torch.mm(features, features.t()) / self.temperature

        # Masks
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        # Samples with same class (excluding self)
        same_class_mask = (labels_col == labels_row).float().fill_diagonal_(0)
        # All other samples (excluding self)
        diff_mask = torch.ones_like(same_class_mask).fill_diagonal_(0)

        # L2C Numerator: exp(proxy) + sum(exp(same_class))
        exp_sim = torch.exp(batch_sim)
        same_class_sum = (exp_sim * same_class_mask).sum(dim=1, keepdim=True)
        numerator = torch.exp(proxy_sim) + same_class_sum

        # L2C Denominator: exp(proxy) + sum(exp(all_others))
        all_sum = (exp_sim * diff_mask).sum(dim=1, keepdim=True)
        denominator = torch.exp(proxy_sim) + all_sum

        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()


# --- Main Model ---


class ContraDKD(nn.Module):
    """
    Unified ContraDKD.

    A single-stage approach where the Student learns via:
    1. DKD (Logits)
    2. Adversarial Alignment (Fooling Discriminator)
    3. Contrastive Alignment (L2C: Clustering features around class proxies)
    """

    def __init__(
        self,
        teacher,
        student,
        teacher_layer,
        student_layer,
        teacher_channels,
        student_channels,
        hidden_channels=256,
        num_classes=10,
        alpha=1.0,  # DKD TCKD
        beta=8.0,  # DKD NCKD
        temperature=4.0,  # DKD Temp
        l2c_weight=0.5,  # Weight for Contrastive Loss
        adv_weight=0.1,  # Weight for Adversarial Loss
    ):
        super(ContraDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.l2c_weight = l2c_weight
        self.adv_weight = adv_weight

        # Freeze Teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Hooks
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        # Projectors (1x1 Conv)
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # Discriminator
        self.discriminator = FeatureDiscriminator(hidden_channels)

        # L2C Loss Module
        self.l2c_loss_mod = L2CContrastiveLoss(num_classes, hidden_channels)

        self.bce_loss = nn.BCELoss()
        self.training_mode = "student"

        print(
            f"ContraDKD Init: Hidden={hidden_channels}, L2C_W={l2c_weight}, Adv_W={adv_weight}"
        )
        print(
            f"Params: Disc={count_params(self.discriminator)}, L2C_Proxies={count_params(self.l2c_loss_mod)}"
        )

    def initialize_class_embeddings(self, dataloader, device="cuda"):
        """
        Initialize L2C class proxies using the Teacher's features.
        This provides a stable target for the Student to align to.
        """
        print("Initializing ContraDKD Proxies from Teacher...")
        self.teacher.eval()
        # Ensure regressor is on device
        self.teacher_regressor.to(device)
        self.teacher_regressor.eval()

        class_feats = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                _ = self.teacher(inputs)
                t_feat = self.teacher_hooks.features.get(self.teacher_layer)

                # Project -> Pool -> Normalize
                t_proj = self.teacher_regressor(t_feat)
                t_vec = F.adaptive_avg_pool2d(t_proj, 1).flatten(1)
                t_vec = F.normalize(t_vec, dim=1)

                for f, l in zip(t_vec, labels):
                    class_feats[l.item()].append(f.cpu())

                self.teacher_hooks.clear()
                if i >= 50:
                    break  # Use subset for speed

        # Average features per class to set proxies
        for c in range(self.num_classes):
            if len(class_feats[c]) > 0:
                center = torch.stack(class_feats[c]).mean(dim=0)
                self.l2c_loss_mod.class_embeddings.data[c] = center.to(device)

        print("Proxies initialized.")

    def set_training_mode(self, mode):
        self.training_mode = mode
        if mode == "discriminator":
            # Train Discriminator & Teacher Regressor & Proxies
            # Freeze Student path
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.student_regressor.parameters():
                p.requires_grad = False

            # Unfreeze Disc path
            for p in self.discriminator.parameters():
                p.requires_grad = True
            for p in self.teacher_regressor.parameters():
                p.requires_grad = True
            self.l2c_loss_mod.class_embeddings.requires_grad = True

        else:  # Student Mode
            # Train Student & Student Regressor
            # Unfreeze Student path
            for p in self.student.parameters():
                p.requires_grad = True
            for p in self.student_regressor.parameters():
                p.requires_grad = True

            # Freeze Disc path
            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.teacher_regressor.parameters():
                p.requires_grad = False
            self.l2c_loss_mod.class_embeddings.requires_grad = False

    def forward(self, x, targets):
        batch_size = x.size(0)

        # 1. Forward Pass
        with torch.no_grad():
            t_logits = self.teacher(x)
        s_logits = self.student(x)

        # 2. Extract & Project Features
        t_feat = self.teacher_hooks.features.get(self.teacher_layer)
        s_feat = self.student_hooks.features.get(self.student_layer)

        if t_feat is None or s_feat is None:
            raise ValueError("Hooks failed to capture features")

        t_proj = self.teacher_regressor(t_feat)
        s_proj = self.student_regressor(s_feat)

        # 3. Prepare Vectors for L2C (Pool -> Flatten -> Normalize)
        t_vec = F.normalize(F.adaptive_avg_pool2d(t_proj, 1).flatten(1), dim=1)
        s_vec = F.normalize(F.adaptive_avg_pool2d(s_proj, 1).flatten(1), dim=1)

        result = {"teacher_logits": t_logits, "student_logits": s_logits}

        if self.training_mode == "discriminator":
            # --- TRAIN DISCRIMINATOR ---
            # D tries to:
            # 1. Classify Teacher as Real (1)
            # 2. Classify Student as Fake (0)
            # 3. Minimize L2C on Teacher Features (Cluster them properly)

            t_pred = self.discriminator(t_proj)
            s_pred = self.discriminator(s_proj.detach())  # Detach student

            real_lbl = torch.ones(batch_size, 1, device=x.device)
            fake_lbl = torch.zeros(batch_size, 1, device=x.device)

            disc_loss = 0.5 * (
                self.bce_loss(t_pred, real_lbl) + self.bce_loss(s_pred, fake_lbl)
            )

            # L2C on Teacher: Ensures the embedding space is structured by class
            l2c_teacher = self.l2c_loss_mod(t_vec, targets)

            result["discriminator_loss"] = disc_loss.item()
            result["l2c_teacher"] = l2c_teacher.item()

            # Total D Loss
            result["total_disc_loss"] = disc_loss + self.l2c_weight * l2c_teacher

            # Metrics
            acc = ((t_pred > 0.5).float().mean() + (s_pred <= 0.5).float().mean()) / 2
            result["discriminator_accuracy"] = acc.item()

        else:
            # --- TRAIN STUDENT ---
            # S tries to:
            # 1. Minimize DKD Loss (Match Logits)
            # 2. Fool Discriminator (Adversarial)
            # 3. Minimize L2C (Cluster features to same proxies as Teacher)

            # A. DKD
            dkd_loss = self.compute_dkd_loss(s_logits, t_logits, targets)

            # B. Adversarial (Fool D -> Output 1)
            s_pred = self.discriminator(s_proj)
            real_lbl = torch.ones(batch_size, 1, device=x.device)
            adv_loss = self.bce_loss(s_pred, real_lbl)

            # C. L2C on Student
            l2c_student = self.l2c_loss_mod(s_vec, targets)

            result["dkd"] = dkd_loss.item()
            result["adversarial"] = adv_loss.item()
            result["l2c_student"] = l2c_student.item()
            result["fool_rate"] = (s_pred > 0.5).float().mean().item()

            # Total Internal Loss (Trainer adds CE)
            # Return just the unified sum for the 'method_specific_loss' slot
            total_internal = (
                dkd_loss + self.adv_weight * adv_loss + self.l2c_weight * l2c_student
            )
            result["total_student_loss"] = total_internal  # For debug
            result["method_specific_loss"] = total_internal

        self.teacher_hooks.clear()
        self.student_hooks.clear()
        return result

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """Standard DKD implementation."""
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student + 1e-8)
        tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean") * (
            self.temperature**2
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = F.kl_div(
            log_pred_student_part2, pred_teacher_part2, reduction="batchmean"
        ) * (self.temperature**2)
        return self.alpha * tckd_loss + self.beta * nckd_loss

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def get_optimizers(self, student_lr=1e-3, discriminator_lr=1e-4, weight_decay=1e-4):
        """
        Returns two optimizers:
        1. Student Optimizer: Student Net + Student Regressor
        2. Discriminator Optimizer: Discriminator Net + Teacher Regressor + L2C Proxies
        """
        # Student Params
        student_params = list(self.student.parameters()) + list(
            self.student_regressor.parameters()
        )

        # Discriminator Params (Includes L2C Proxies)
        discriminator_params = (
            list(self.discriminator.parameters())
            + list(self.teacher_regressor.parameters())
            + list(self.l2c_loss_mod.parameters())
        )

        opt_s = torch.optim.Adam(
            student_params, lr=student_lr, weight_decay=weight_decay
        )
        opt_d = torch.optim.Adam(
            discriminator_params, lr=discriminator_lr, weight_decay=weight_decay
        )

        return opt_s, opt_d
