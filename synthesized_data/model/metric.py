import torch
import numpy as np

def metric_overall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def metric_noisy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        idxs = np.where((target < 5) == True)[0]
        correct += np.sum(pred[idxs] == target[idxs])
    return correct / len(idxs)

def metric_control(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        idxs = np.where((target >= 5) == True)[0]
        correct += np.sum(pred[idxs] == target[idxs])
    return correct / len(idxs)

def metric_top5(output, target, k=5):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
