from dataset import SegData, SegDataTest
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model import UNet
from torch_snippets import *
import warnings
warnings.filterwarnings("ignore")
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Определяем тестовый и тренировочный датасеты и аналогично даталоудеры:
trn_ds = SegData('train')
val_ds = SegData('test')
trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)
TEST_ds = SegDataTest('TEST')
TEST_dl = DataLoader(TEST_ds, batch_size=1,collate_fn=TEST_ds.collate_fn)

# Определим функцию потерь
ce = nn.CrossEntropyLoss()

def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc

# Функции для тренировки на батче данных и расчет метрик для валидациии:
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

with torch.no_grad():
    def validate_batch(model, data, criterion):
        model.eval()
        ims, masks = data
        _masks = model(ims)
        loss, acc = criterion(_masks, masks)
        return loss.item(), acc.item()

model = UNet().to(device)
criterion = UnetLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 10

log = Report(n_epochs)
for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model, data, optimizer,criterion)
        log.record(ex+(bx+1)/N, trn_loss=loss,trn_acc=acc, end='\r')

    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss, acc = validate_batch(model, data, criterion)
        log.record(ex+(bx+1)/N, val_loss=loss,val_acc=acc, end='\r')

    log.report_avgs(ex+1)
