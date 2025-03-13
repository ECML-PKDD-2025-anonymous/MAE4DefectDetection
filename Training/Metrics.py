import torch
import torch.nn as nn
from piqa import SSIM, PSNR

def get_evaluation_metrics(y_true, y_pred, device):

    # Set up evaluation metrics on device 
    mse_loss = nn.MSELoss().to(device)
    psnr = PSNR().to(device)
    ssim = SSIM(n_channels=1, window_size=11, value_range=1).to(device)

    # Calculate evaluation metrics 
    mse = mse_loss(y_true, y_pred)
    structural_similarity = ssim(y_true, y_pred)
    peak_signal_to_noise = psnr(y_true, y_pred)
    
    return mse, structural_similarity, peak_signal_to_noise

def get_accuracy(targets, predictions, device):
    
    # Move predictions and targets to device
    predictions = predictions.to(device)
    targets = targets.to(device)

    # Get the predicted class by taking the index of the max logit for each example (Top-1 prediction)
    _, predicted_classes = torch.max(predictions, dim=1)

    target_classes = []
    # Get indexes from one-hot-encoded targets
    for one_hot_vector in targets:
        for class_id in range(one_hot_vector.size(0)):
            if one_hot_vector[class_id] == 1:
                target_classes.append(class_id)
    target_classes = torch.Tensor(target_classes).to(device)

    # Compare predicted classes with the actual targets
    correct = (target_classes == predicted_classes).sum().item()

    # Calculate accuracy as the number of correct predictions divided by batch size
    accuracy = (correct / targets.size(0)) * 100

    return accuracy

def get_segmentation_metrics(y_true, y_pred):
    prec = precision(y_true, y_pred) 
    rec = recall(y_true, y_pred) 
    iou = iou_score(y_true, y_pred) 
    dice = dice_score(y_true, y_pred) 
    return {'precision': prec, 'recall': rec, 'iou': iou, 'dice': dice}

def precision(y_true, y_pred):
    predicted_positives = torch.sum(y_pred)
    true_positives = (y_true * y_pred).sum()
    return true_positives / predicted_positives.clamp(min=1)


def recall(y_true, y_pred):
    actual_positives = torch.sum(y_true)
    true_positives = (y_true * y_pred).sum()
    return true_positives / actual_positives.clamp(min=1)


def iou_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = ((y_true + y_pred) >= 1).float().sum()
    return intersection / union.clamp(min=1e-6)

def miou_score(y_true, y_pred):
    sum = 0 
    for i in range(y_true.shape[0]):
        true = y_true[i]
        pred = y_pred[i]
        intersection = (true * pred).sum()
        union = ((true + pred) >= 1).float().sum()
        sum += intersection / union.clamp(min=1e-6)
    return sum / y_true.shape[0]


def dice_score(y_true, y_pred):
    true_positives = (y_true * y_pred).sum()
    false_positives = ((1 - y_true) * y_pred).sum()
    false_negatives = (y_true * (1 - y_pred)).sum()
    return (2. * true_positives) / (2. * true_positives + false_positives + false_negatives).clamp(min=1e-6)