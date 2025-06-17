import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from peft import LoraConfig, TaskType
from peft import LoraModel

from src.utils import EarlyStoppingWithWarmup, train, evaluate
from src.model import DinoVisionTransformerClassifier
from src.dataset import ImageFolderDataModule

## Data loader
dm = ImageFolderDataModule(root_dir='./data/FGVC_aircraft', batch_size=64)
dm.setup()

train_loader, val_loader, test_loader = dm.get_loaders()

device='cuda'

## Model
model = DinoVisionTransformerClassifier(n_class = dm.class_num)
model = model.to(device)
model.config = {
    "model_type": "dino"
}

## Learning
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    verbose=True,
)
early_stopping = EarlyStoppingWithWarmup(monitor="val_acc", mode="max", patience=5, warmup=5)

# PEFT
peft_config = LoraConfig(
    r=8,
    lora_alpha=1.0,
    lora_dropout=0,
    target_modules=['qkv'],
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
    # DoRA applying
    use_dora=True,
)
LoraModel(model, config=peft_config, adapter_name='default')

for name, param in model.named_parameters() :
    if 'classifier' in name :
        param.requires_grad_(True)

## Train
for epoch in range(100):
    print(f"\nEpoch {epoch+1}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Val   Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_acc)
    early_stopping.step(val_acc)

    if early_stopping.stop_training:
        break

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Acc : {test_acc:.4f}")