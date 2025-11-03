# WH-DisDKD
# Knowledge Distillation Toolbox

Compact, easy-to-follow training repo for experimenting with multiple knowledge-distillation (KD) methods — logit-based, feature-based, contrastive, and a discriminator-guided variant.

---

## Supported methods
The code exposes the `--method` flag. Available choices:
- `Pretraining` — train student from data only (no KD).
- `LogitKD` — vanilla logit distillation (Hinton et al.).  
  https://arxiv.org/abs/1503.02531
- `DKD` — Decoupled Knowledge Distillation (Zhao et al.).  
  https://arxiv.org/abs/2203.08679
- `DisDKD` — our Discriminator-guided feature alignment + DKD at logits (custom).
- `FitNet` — FitNets feature distillation (Romero et al.).  
  https://arxiv.org/abs/1412.6550
- `CRD` — Contrastive Representation Distillation (Tian et al.).  
  https://arxiv.org/abs/1910.10699

> Note: the flag string must match exactly one of the choices above (see `config.py`).

---

## Quick start
1. Running LogitKD method:
'''bash 
python train.py \
  --method LogitKD \
  --teacher resnet50 \
  --student resnet18 \
  --dataset CIFAR100 \
  --batch_size 128 \
  --epochs 100 \
  --lr 0.01 \
  --tau 4.0 \
  --alpha 1.0 --beta 0.4 \
  --save_dir ./checkpoints/logitkd
  '''
  2. Running DisDKD (our custom method)
  '''bash
  python train.py \
  --method DisDKD \
  --teacher resnet50 \
  --student resnet18 \
  --dataset IMAGENETTE \
  --epochs 60 \
  --disdkd_adversarial_weight 0.01 \
  --discriminator_lr 1e-4 \
  --disc_lr_multiplier 1.0 \
  --dkd_alpha 1.0 --dkd_beta 8.0 \
  --save_dir ./checkpoints/disdkd
  '''


