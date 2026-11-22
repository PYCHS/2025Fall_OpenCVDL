"""Compare LeNet activations with validation-only checkpoint selection."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision
from model import LeNet5


def make_loaders(data_dir, batch_size, seed):
    evaluation = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    augmentation = transforms.Compose([
        transforms.Resize((32, 32)), transforms.RandomRotation(15),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    training_data = datasets.MNIST(data_dir, train=True, download=True, transform=augmentation)
    validation_data = datasets.MNIST(data_dir, train=True, download=True, transform=evaluation)
    indices = torch.randperm(len(training_data), generator=torch.Generator().manual_seed(seed))
    train_indices, val_indices = indices[:55000], indices[55000:]
    assert len(set(train_indices.tolist()) & set(val_indices.tolist())) == 0
    assert len(train_indices) + len(val_indices) == 60000
    train = DataLoader(Subset(training_data, train_indices.tolist()), batch_size=batch_size,
                       shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(Subset(validation_data, val_indices.tolist()), batch_size=batch_size)
    metadata = {
        'train_count': len(train_indices), 'validation_count': len(val_indices),
        'split_seed': seed,
        'train_indices_sha256': hashlib.sha256(train_indices.numpy().tobytes()).hexdigest(),
        'validation_indices_sha256': hashlib.sha256(val_indices.numpy().tobytes()).hexdigest(),
        'train_augmentation': 'Resize(32,32), RandomRotation(15), ToTensor, Normalize(0.5,0.5)',
        'evaluation_transform': 'Resize(32,32), ToTensor, Normalize(0.5,0.5)',
    }
    return train, validation, evaluation, metadata


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item() * labels.numel()
            correct += (logits.detach().argmax(1) == labels).sum().item()
            count += labels.numel()
    return {'loss': total_loss / count, 'accuracy_percent': 100 * correct / count,
            'correct': correct, 'count': count}


def train_process(activation, args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    train, validation, evaluation, split = make_loaders(args.data_dir, args.batch_size, args.seed)
    model = LeNet5(activation=activation.lower()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint = args.output_dir / f'Weight_{activation}.pth'
    history, best_accuracy, best_epoch = [], -1.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train, device, optimizer)
        val_metrics = run_epoch(model, validation, device)
        history.append({'epoch': epoch, 'train': train_metrics, 'validation': val_metrics})
        if val_metrics['accuracy_percent'] > best_accuracy:
            best_accuracy, best_epoch = val_metrics['accuracy_percent'], epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
        print(f'{activation} epoch {epoch}/{args.epochs} '
              f'train={train_metrics["accuracy_percent"]:.2f}% '
              f'validation={val_metrics["accuracy_percent"]:.2f}%', flush=True)
    training_seconds = time.perf_counter() - started
    # The official test set is created and evaluated only after model selection ends.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=evaluation)
    test = run_epoch(model, DataLoader(test_data, batch_size=args.batch_size), device)
    result = {
        'activation': activation, 'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'optimizer': 'Adam', 'split': split, 'best_epoch': best_epoch,
        'best_validation_accuracy_percent': best_accuracy, 'test': test,
        'selection_rule': 'highest validation accuracy; earliest epoch on ties',
        'test_evaluations': 1, 'training_seconds': training_seconds,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'device': str(device),
                        'platform': platform.platform(), 'threads': torch.get_num_threads()},
        'history': history,
        'limitations': ['Single seed per activation; no statistical superiority claim.',
                        'Historical repository checkpoints used test-based selection; these are retrained from scratch.'],
    }
    (args.output_dir / f'results_{activation}.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split_name in ['train', 'validation']:
        axes[0].plot(range(1, args.epochs + 1), [r[split_name]['loss'] for r in history], label=split_name)
        axes[1].plot(range(1, args.epochs + 1), [r[split_name]['accuracy_percent'] for r in history], label=split_name)
    for ax, title in zip(axes, ['Cross-entropy loss', 'Accuracy (%)']):
        ax.set(xlabel='Epoch', title=f'{activation}: {title}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f'Loss&Acc_{activation}.jpg', dpi=150)
    plt.close(fig)
    print(f'{activation}: selected epoch {best_epoch}; test {test["correct"]}/{test["count"]} '
          f'({test["accuracy_percent"]:.2f}%)', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--activations', choices=['Sigmoid', 'Relu'], nargs='+', default=['Sigmoid', 'Relu'])
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'runs' / 'validation_split')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.threads < 1:
        parser.error('epochs, batch size, learning rate, and threads must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve existing experiment records')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    # CPU is the portable default; MPS is opt-in and recorded in experiment metadata.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    results = [train_process(activation, args, device) for activation in args.activations]
    (args.output_dir / 'summary.json').write_text(json.dumps([
        {k: r[k] for k in ['activation', 'seed', 'best_epoch', 'best_validation_accuracy_percent', 'test']}
        for r in results], indent=2) + '\n')


if __name__ == '__main__':
    main()
