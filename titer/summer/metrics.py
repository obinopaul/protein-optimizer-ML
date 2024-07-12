import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

class Specificity(nn.Module):
    """This class provides the specificity metric"""
    def __init__(self):
        super(Specificity, self).__init__()

    def forward(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        return specificity

class Sensitivity(nn.Module):
    """This class provides the Sensitivity metric"""
    def __init__(self):
        super(Sensitivity, self).__init__()

    def forward(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        return sensitivity

class GeometricMean(nn.Module):
    """This class provides the G-mean metric, which can be used for model selection"""
    def __init__(self):
        super(GeometricMean, self).__init__()

    def forward(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        gmean = np.sqrt(specificity * sensitivity)
        return gmean

class EarlyDetectionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_arl(self):
        # Calculate Average Run Length (ARL)
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        arl = (tp + fn) / (tp + fn)
        return arl

    def calculate_arlidx(self):
        # Calculate ARL Index (ARLIDX)
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        arlidx = (fp + tn) / (fp + tn)
        return arlidx

    def calculate_usp(self):
        # Calculate Percentage of Undetected Samples (USP)
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        usp = fn / (tp + fn)
        return usp

    def calculate_tasrid(self):
        # Calculate True Alert Streaks Rate from Initial Detection (TASRID)
        true_alert_streaks = []
        for i in range(len(self.y_true)):
            if self.y_true[i] == self.y_pred[i] == 1:
                true_alert_streaks.append(1)
            else:
                true_alert_streaks.append(0)
        tasrid = sum(true_alert_streaks) / len(true_alert_streaks)
        return tasrid

def calculate_metrics(y_true, y_pred):
    specificity = Specificity()(y_true, y_pred)
    sensitivity = Sensitivity()(y_true, y_pred)
    gmean = GeometricMean()(y_true, y_pred)
    
    early_detection_metrics = EarlyDetectionMetrics(y_true, y_pred)
    arl = early_detection_metrics.calculate_arl()
    arlidx = early_detection_metrics.calculate_arlidx()
    usp = early_detection_metrics.calculate_usp()
    tasrid = early_detection_metrics.calculate_tasrid()
    
    metrics = {
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'G-mean': gmean,
        'ARL': arl,
        'ARLIDX': arlidx,
        'USP': usp,
        'TASRID': tasrid
    }
    
    return metrics
