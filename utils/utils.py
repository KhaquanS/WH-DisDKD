import torch
from utils.data import DOMAINBED_DATASETS

def list_all_datasets():
    """List all available DomainBed datasets with their domains."""
    print("=== Available DomainBed Datasets ===")
    for dataset_name, info in DOMAINBED_DATASETS.items():
        print(f"\n{dataset_name}:")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Domains: {info['domains']}")
    print("\n" + "=" * 40)


def suggest_domain_splits(dataset_name: str):
    """Suggest common domain splits for a dataset."""
    dataset_name = dataset_name.upper()
    
    if dataset_name not in DOMAINBED_DATASETS:
        print(f"Dataset {dataset_name} not found in DomainBed datasets.")
        return
    
    domains = DOMAINBED_DATASETS[dataset_name]['domains']
    print(f"\n=== Domain Split Suggestions for {dataset_name} ===")
    
    # Common split patterns
    splits = [
        {
            'name': 'Leave-one-out (recommended)',
            'description': 'Use 3 domains for training, 1 for validation',
            'splits': [(domains[:-1], domains[-1])]
        },
        {
            'name': 'Single domain training',
            'description': 'Train on 1 domain, validate on another',
            'splits': [([domains[0]], domains[1]), ([domains[1]], domains[0])]
        },
        {
            'name': 'Two-domain training',
            'description': 'Train on 2 domains, validate on 1',
            'splits': [(domains[:2], domains[2]), (domains[1:3], domains[0])]
        }
    ]
    
    for split_type in splits:
        print(f"\n{split_type['name']}:")
        print(f"  {split_type['description']}")
        for i, (train_doms, val_dom) in enumerate(split_type['splits']):
            train_str = ','.join(train_doms)
            print(f"  Option {i+1}: --train_domains \"{train_str}\" --val_domain \"{val_dom}\"")
    
    print("\n" + "=" * 50)


def validate_domain_arguments(dataset_name: str, train_domains_str: str, val_domain: str):
    """Validate domain arguments and provide feedback."""
    dataset_name = dataset_name.upper()
    
    if dataset_name not in DOMAINBED_DATASETS:
        return True  # Not a DomainBed dataset
    
    available_domains = DOMAINBED_DATASETS[dataset_name]['domains']
    
    # Parse training domains
    if train_domains_str:
        train_domains = [d.strip() for d in train_domains_str.split(',')]
    else:
        train_domains = []
    
    # Check validity
    invalid_train = [d for d in train_domains if d not in available_domains]
    if invalid_train:
        print(f"Error: Invalid training domains for {dataset_name}: {invalid_train}")
        print(f"Available domains: {available_domains}")
        return False
    
    if val_domain and val_domain not in available_domains:
        print(f"Error: Invalid validation domain '{val_domain}' for {dataset_name}")
        print(f"Available domains: {available_domains}")
        return False
    
    # Check for overlap
    if val_domain and val_domain in train_domains:
        print(f"Warning: Validation domain '{val_domain}' is also in training domains.")
        print("This creates data leakage and should be avoided for proper OOD evaluation.")
    
    return True


def generate_experiment_commands(dataset_name: str, method: str = 'LogitKD'):
    """Generate example command lines for different domain splits."""
    dataset_name = dataset_name.upper()
    
    if dataset_name not in DOMAINBED_DATASETS:
        print(f"Dataset {dataset_name} not found in DomainBed datasets.")
        return
    
    domains = DOMAINBED_DATASETS[dataset_name]['domains']
    
    print(f"\n=== Example Commands for {dataset_name} ===")
    
    # Leave-one-out experiments
    print(f"\n# Leave-one-out experiments:")
    for i, val_domain in enumerate(domains):
        train_domains = [d for d in domains if d != val_domain]
        train_str = ','.join(train_domains)
        
        cmd = (f"python main.py --dataset {dataset_name} --method {method} "
               f"--train_domains \"{train_str}\" --val_domain \"{val_domain}\" "
               f"--epochs 100 --batch_size 32")
        print(f"# Experiment {i+1}: Train on {train_domains}, validate on {val_domain}")
        print(cmd)
        print()
    
    print("=" * 60)


def check_domainbed_installation():
    """Check if DomainBed is properly installed."""
    try:
        import DomainBed.domainbed
        print("✓ DomainBed is installed and available")
        return True
    except ImportError:
        print("✗ DomainBed is not installed")
        print("\nTo install DomainBed:")
        print("  git clone https://github.com/facebookresearch/DomainBed")
        print("  cd DomainBed")
        print("  pip install -e .")
        print("\nOr install from PyPI:")
        print("  pip install domainbed")
        return False


def get_module(model, module_name):
    """
    Get a module from a model by its name (dot-separated path).
    
    Args:
        model (nn.Module): The model to extract the module from
        module_name (str): Dot-separated path to the module (e.g., 'layer1.0.conv1')
        
    Returns:
        nn.Module: The requested module
    """
    modules = module_name.split('.')
    current_module = model
    
    for module in modules:
        # Handle integer indices for sequential containers
        if module.isdigit():
            current_module = current_module[int(module)]
        else:
            current_module = getattr(current_module, module)
    
    return current_module


def count_params(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): The model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output (Tensor): Model predictions
        target (Tensor): Ground truth labels
        topk (tuple): Which top-k accuracies to compute
        
    Returns:
        list: Accuracy values for each k in topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='best.pth'):
    """
    Save model checkpoint.
    
    Args:
        state (dict): Checkpoint state dictionary
        is_best (bool): Whether this is the best model so far
        filename (str): Filename for regular checkpoint
        best_filename (str): Filename for best checkpoint
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        
    Returns:
        dict: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma=0.1):
    """
    Adjust learning rate based on schedule.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to adjust
        epoch (int): Current epoch
        lr (float): Base learning rate
        schedule (list): List of epochs to decay learning rate
        gamma (float): Multiplicative factor of learning rate decay
    """
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return lr


class WarmupLRScheduler:
    """
    Learning rate scheduler with warmup.
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.epoch = 0
    
    def step(self):
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr + (self.max_lr - self.base_lr) * self.epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.epoch += 1


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False