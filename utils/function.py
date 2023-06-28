from tqdm import tqdm
from collections import OrderedDict
import torchmetrics
from torchmetrics.classification import MulticlassAveragePrecision
import torch

# =======================train======================== 
def train(train_loader, net, criterion1, criterion2, optimizer, device):
    net.train()
    size = len(train_loader.dataset)

    t = tqdm(enumerate(train_loader), total=len(train_loader), ncols=80)
    for batch_idx, (inputs, targets) in t:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        preds = net(inputs)

        loss = criterion1(preds, targets)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            t.set_postfix(loss=loss.item())
        # # 每训练100次，输出一次当前信息
        # if batch_idx % 10 == 0:
        #     loss, current = loss.item(), batch_idx * len(inputs)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# =======================test======================== 
def test(dataloader, model, criterion, device, args):
    size = len(dataloader.dataset)

    test_loss, correct = 0, 0

    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=args.classes).to(device)
    test_recall = torchmetrics.Recall(task='multiclass',average='none', num_classes=args.classes).to(device)
    test_precision = torchmetrics.Precision(task='multiclass',average='none', num_classes=args.classes).to(device)
    test_ap = MulticlassAveragePrecision(num_classes=args.classes)
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            # test_loss += criterion(preds, targets).item()

            correct += (preds.argmax(1) == targets).type(torch.float).sum().item()

            test_recall(preds.argmax(1), targets)
            test_precision(preds.argmax(1), targets)
            test_ap(preds, targets)
            test_acc(preds.argmax(1), targets)

    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_acc = test_acc.compute()
    mAP = test_ap.compute()

    print("pre:", total_precision)
    print("sen:", total_recall)
    print("mAP:", mAP.item())
    
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()

    test_loss /= size
    correct /= size
    print("correct = ", correct)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    return correct
