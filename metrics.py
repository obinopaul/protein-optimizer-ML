from scipy.integrate import simpson
import warnings
import inspect
import itertools
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, auc, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, roc_curve, recall_score
from sklearn.model_selection import learning_curve, cross_val_score, KFold, train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,  recall_score, f1_score,
    average_precision_score, roc_curve, auc,
    )
from sklearn.model_selection import (
    validation_curve, learning_curve, cross_val_score, KFold, train_test_split
    )
from yellowbrick.model_selection import CVScores
from yellowbrick.classifier import ClassPredictionError

def true_positives(y_true, y_pred):
    tp = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 1 and label == 1:
            tp += 1
    return tp


def true_negatives(y_true, y_pred):
    tn = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 0 and label == 0:
            tn += 1
    return tn


def false_positives(y_true, y_pred):
    fp = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 1 and label == 0:
            fp += 1
    return fp


def false_negatives(y_true, y_pred):
    fn = 0
    for label, pred in zip(y_true, y_pred):
        if pred == 0 and label == 1:
            fn += 1
    return fn

def false_positives(X_test, y_true, y_pred, classes):
    """ 
    This function identifies and plots the false positives in a classification problem. 
    """ 
    fp_indices = np.where((y_true != class_of_interest) & (y_pred == class_of_interest))[0] 
    fp_features = X_test[fp_indices] # assuming X_test is a numpy array of input data 
    # fp_features = X_test.iloc[fp_indices]
    fp_labels = y_pred[fp_indices] # assuming y_pred is a numpy array of predicted labels 
    # fp_labels = pd.Series(y_pred).iloc[fp_indices]

    print("False positives: ", len(fp_indices))
    return fp_features, fp_labels


#false negatives 
def false_negatives(X_test, y_true, y_pred, classes):
    """ 
    This function identifies and plots the false negatives in a classification problem. 
    """ 
    fn_indices = np.where((y_true == class_of_interest) & (y_pred != class_of_interest))[0] 
    fn_features = X_test[fn_indices] # assuming X_test is a numpy array of input data
    # fn_features = X_test.iloc[fn_indices] 
    fn_labels = y_pred[fn_indices] # assuming y_pred is a numpy array of predicted labels 
    # fn_labels = pd.Series(y_pred).iloc[fn_indices]

    print("False negatives: ", len(fn_indices))
    return fn_features, fn_labels

def binary_accuracy(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(y_true, y_pred):
    """
    Fraction of True Positive Elements divided by total number of positive predicted units
    How I view it: Assuming we say someone has cancer: how often are we correct?
    It tells us how much we can trust the model when it predicts an individual as positive.
    """
    tp = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return tp / (tp + fp)


def recall(y_true, y_pred):
    """
    Recall meaasure the model's predictive accuracy for the positive class.
    How I view it, out of all the people that has cancer: how often are
    we able to detect it?
    """
    tp = true_negatives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return tp / (tp + fn)


def multiclass_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for label, pred in zip(y_true, y_pred):
        correct += label == pred
    return correct/total


def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_true.shape == y_pred.shape
    unique_classes = np.unique(np.concatenate([y_true, y_pred], axis=0)).shape[0]
    cm = np.zeros((unique_classes, unique_classes), dtype=np.int64)

    for label, pred in zip(y_true, y_pred):
        cm[label, pred] += 1

    return cm


def accuracy_cm(cm):
    return np.trace(cm)/np.sum(cm)


def balanced_accuracy_cm(cm):
    correctly_classified = np.diagonal(cm)
    rows_sum = np.sum(cm, axis=1)
    indices = np.nonzero(rows_sum)[0]
    if rows_sum.shape[0] != indices.shape[0]:
        warnings.warn("y_pred contains classes not in y_true")
    accuracy_per_class = correctly_classified[indices]/(rows_sum[indices])
    return np.sum(accuracy_per_class)/accuracy_per_class.shape[0]


def precision_cm(cm, average="specific", class_label=1, eps=1e-12):
    tp = np.diagonal(cm)
    fp = np.sum(cm, axis=0) - tp
    #precisions = np.diagonal(cm)/np.maximum(np.sum(cm, axis=0), 1e-12)

    if average == "none":
        return tp/(tp+fp+eps)

    if average == "specific":
        precisions = tp / (tp + fp + eps)
        return precisions[class_label]

    if average == "micro":
        # all samples equally contribute to the average,
        # hence there is a distinction between highly
        # and poorly populated classes
        return np.sum(tp) / (np.sum(tp) + np.sum(fp) + eps)

    if average == "macro":
        # all classes equally contribute to the average,
        # no distinction between highly and poorly populated classes.
        precisions = tp / (tp + fp + eps)
        return np.sum(precisions)/precisions.shape[0]

    if average == "weighted":
        pass


def recall_cm(cm, average="specific", class_label=1, eps=1e-12):
    tp = np.diagonal(cm)
    fn = np.sum(cm, axis=1) - tp

    if average == "none":
        return tp / (tp + fn + eps)

    if average == "specific":
        recalls = tp / (tp + fn + eps)
        return recalls[class_label]

    if average == "micro":
        return np.sum(tp) / (np.sum(tp) + np.sum(fn))

    if average == "macro":
        recalls = tp / (tp + fn + eps)
        return np.sum(recalls)/recalls.shape[0]

    if average == "weighted":
        pass


def f1score_cm(cm, average="specific", class_label=1):
    precision = precision_cm(cm, average, class_label)
    recall = recall_cm(cm, average, class_label)
    return 2 * (precision*recall)/(precision+recall)


# --> REGRESSION METRICS


def roc_curve(y_true, y_preds, plot_graph=True, calculate_AUC=True, threshold_step=0.01):
    TPR, FPR = [], []

    for threshold in np.arange(np.min(y_preds), np.max(y_preds), threshold_step):
        predictions = (y_preds > threshold) * 1
        cm = confusion_matrix(y_true, predictions)
        recalls = recall_cm(cm, average="none")
        # note TPR == sensitivity == recall
        tpr = recalls[1]
        # note tnr == specificity (which is same as recall for the negative class)
        tnr = recalls[0]
        TPR.append(tpr)
        FPR.append(1-tnr)

    if plot_graph:
        plt.plot(FPR, TPR)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.show()

    if calculate_AUC:
        print(np.abs(np.trapz(TPR, FPR)))


def precision_recall_curve(y_true, y_preds, plot_graph=True, calculate_AUC=True, threshold_step=0.01):
    recalls, precisions = [], []

    for threshold in np.arange(np.min(y_preds), np.max(y_preds), threshold_step):
        predictions = (y_preds > threshold) * 1
        cm = confusion_matrix(y_true, predictions)
        recall = recall_cm(cm, average="specific", class_label=1)
        precision = precision_cm(cm, average="specific", class_label=1)
        recalls.append(recall)
        precisions.append(precision)

    recalls.append(0)
    precisions.append(1)

    if plot_graph:
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.show()

    if calculate_AUC:
        print(np.abs(np.trapz(precisions, recalls)))




########### Model Explanation ###########
## Plotting AUC ROC curve
def plot_roc(y_actual, y_pred):
    """
    Function to plot AUC-ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    plt.plot(
        fpr,
        tpr,
        color="b",
        label=r"Model (AUC = %0.2f)" % (roc_auc_score(y_actual, y_pred)),
        lw=2,
        alpha=0.8,
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Luck (AUC = 0.5)",
        alpha=0.8,
    )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def plot_precisionrecall(y_actual, y_pred):
    """
    Function to plot AUC-ROC curve
    """
    average_precision = average_precision_score(y_actual, y_pred)
    precision, recall, _ = precision_recall_curve(y_actual, y_pred)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = (
        {"step": "post"} if "step" in inspect.signature(plt.fill_between).parameters else {}
    )

    plt.figure(figsize=(9, 6))
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve: AP={0:0.2f}".format(average_precision))


## Plotting confusion matrix
def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


## Variable Importance plot
def feature_importance(model, X):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()
    


def print_classification_performance2class_report(model,X_test,y_test):
    """ 
        Program: print_classification_performance2class_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report
    """
    import seaborn as sns
    
    sns.set()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    TN, FP, FN, TP = conf_mat.ravel()
    PC = precision_score(y_test, y_pred, zero_division=0)
    RC = recall_score(y_test, y_pred, zero_division=0)
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    FS = f1_score(y_test, y_pred)
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    gmean = np.sqrt(RC * specificity) if RC * specificity >= 0 else 0
    
    # Calculate ROC curve and AUC
    pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    roc_auc = auc(pfr, tpr)
    
    print("Accuracy: {:.2%}".format(ACC))
    print("Precision: {:.2%}".format(PC))
    print("Sensitivity (Recall): {:.2%}".format(RC))
    print("Specificity: {:.2%}".format(specificity))
    print("Fscore: {:.2%}".format(FS))
    print("Average precision: {:.2%}".format(AP))
    print("G-Mean: {:.2%}".format(gmean))
    print("ROC AUC: {:.2%}".format(roc_auc))

    
    fig = plt.figure(figsize=(20,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    
    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)
    
    #ROC
    plt.subplot(142)
    # pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    # roc_auc = auc(pfr, tpr)
    gini = (roc_auc*2)-1
    plt.plot(pfr, tpr, label='ROC Curve (area =  {:.2%})'.format(roc_auc) )
    plt.plot([0,1], [0,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Charecteristic Curve with Gini {:.2}'.format(gini))
    plt.legend(loc='lower right')
    
    #pr
    plt.subplot(143)
    precision, recall, _ = precision_recall_curve(y_test,y_pred_proba)
    step_kwargs = ({'step':'post'}
                  if 'step'in inspect.signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2, where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('2-class Precision-Recall Curve: AP={:.2%}'.format(AP))
    
    #hist
    plt.subplot(144)
    tmp = pd.DataFrame(data=[y_test,y_pred_proba]).transpose()
    tmp.columns=['class','proba']
    mask_c0 = tmp['class']==0
    mask_c1 = tmp['class']==1
    plt.hist(tmp.loc[mask_c0,'proba'].dropna(),density=True,alpha=0.5,label='0',bins=20)
    plt.hist(tmp.loc[mask_c1,'proba'].dropna(),density=True,alpha=0.5,label='1',bins=20)
    plt.ylabel('Density')
    plt.xlabel('Probability')
    plt.title('2-class Distribution' )
    plt.legend(loc='upper right')
    
    plt.show()
    
    return y_pred,ACC,PC,RC,FS,AP,roc_auc,gini



def print_classification_performance_multiclass_report(model, X_test, y_test):
    """
    Function to print standard classification metrics report for multiclass classification.
    """
    sns.set()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Accuracy
    ACC = accuracy_score(y_test, y_pred)
    
    # Precision, Recall, F1-Score
    # Note: To calculate average precision for multiclass, a "One-vs-Rest" approach is typically used.
    PC = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    RC = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    FS = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC AUC Calculation for Multiclass
    # Note: This requires handling probabilities for each class separately
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    print("Accuracy: {:.2%}".format(ACC))
    print("Precision: {:.2%}".format(PC))
    print("Recall: {:.2%}".format(RC))
    print("F1 Score: {:.2%}".format(FS))
    print("ROC AUC (One-vs-Rest): {:.2%}".format(roc_auc))

    fig = plt.figure(figsize=(12, 6))
    
    # Confusion Matrix Heatmap
    plt.subplot(121)
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # ROC Curve - One-vs-Rest approach
    plt.subplot(122)
    fpr = dict()
    tpr = dict()
    for i in range(len(model.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {auc(fpr[i], tpr[i]):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # Return performance metrics
    return y_pred, ACC, PC, RC, FS, roc_auc


def plot_predictions(y_pred, y_test): 
    """
    Plots the predicted and actual values on separate scatter plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the actual values
    ax1.scatter(range(len(y_test)), y_test, label='Actual Values')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Scatter plot of Actual Values')
    ax1.legend()
    
    # Plot the predicted values
    ax2.scatter(range(len(y_pred)), y_pred, label='Predicted Values')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Scatter plot of Predicted Values')
    ax2.legend()
    
    # Show the plots
    plt.show()



def check_imbalance(dataset, columns=None, threshold=10):
    """
    This function takes a dataset and one or more columns as input and returns True if any of the specified columns
    are imbalanced, False otherwise. A column is considered imbalanced if the percentage of the minority class is less
    than the specified threshold.
    """
    # If no columns are specified, use all columns except for the last one as the features
    if columns is None:
        features = dataset.iloc[:, :-1]
        columns = features.columns
    
    # Check the imbalance of each specified column
    for col in columns:
        # Get the counts of each class in the column
        class_counts = dataset[col].value_counts()

        # Calculate the percentage of each class in the column
        class_percentages = class_counts / len(dataset) * 100

        # Plot the class percentages
        plt.bar(class_counts.index, class_percentages)
        plt.xlabel(col)
        plt.ylabel('Percentage')
        plt.title(f'{col} Distribution')
        plt.show()

        # Check if the column is imbalanced
        minority_class = class_counts.index[-1]
        minority_class_percentage = class_percentages.iloc[-1]
        if minority_class_percentage < threshold:
            print(f'{col} is imbalanced. Minority class: {minority_class}, Percentage: {minority_class_percentage:.2f}%')
            return True

    # If none of the specified columns are imbalanced, return False
    print('No imbalance found.')
    return False


#if there is imbalance, you can handle it by over-sampling or under-sampling the dataset

def handle_imbalanced_data(X, y, strategy='over-sampling'):
    """
    Handle imbalanced data using imblearn library.
    
    Parameters:
    -----------
    X: array-like of shape (n_samples, n_features)
        The input data.
    y: array-like of shape (n_samples,)
        The target values.
    strategy: str, default='over-sampling'
        The strategy to use for handling imbalanced data. Possible values are
        'over-sampling' and 'under-sampling'.
        
    Returns:
    --------
    X_resampled: array-like of shape (n_samples_new, n_features)
        The resampled input data.
    y_resampled: array-like of shape (n_samples_new,)
        The resampled target values.
    """
    if strategy == 'over-sampling':
        # Initialize the RandomOverSampler object
        ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
        # Resample the data
        X_resampled, y_resampled = ros.fit_resample(X, y)
    elif strategy == 'under-sampling':
        # Initialize the RandomUnderSampler object
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=0)
        # Resample the data
        X_resampled, y_resampled = rus.fit_resample(X, y)
    else:
        raise ValueError("Invalid strategy. Possible values are 'over-sampling' and 'under-sampling'.")
    
    return X_resampled, y_resampled


def cv_learning_curve(model, X, y, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy')
                                                #scoring parameter -  #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
    train_mean = np.mean(-train_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    test_mean = np.mean(-test_scores, axis=1)
    test_std = np.std(-test_scores, axis=1)
    
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Error')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation Error')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()
    
    return train_sizes, train_mean, train_std, test_mean, test_std

def cv_bias_variance(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')   
    train_error = -scores.mean()
    val_error = -scores.std()
    return train_error, val_error, scores


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

def print_regression_performance_report(model, X_test, y_test):
    """ 
    Program: print_regression_performance_report
    Author: Siraprapa W.
    
    Purpose: Print standard regression metrics report and generate visualizations.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print('Mean Squared Error (MSE): {:.4f}'.format(mse))
    print('Root Mean Squared Error (RMSE): {:.4f}'.format(rmse))
    print('R-squared (R2): {:.4f}'.format(r2))
    
    # Add additional regression metrics and visualizations here
    
    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
    
    return mse, rmse, r2, y_pred

def evaluate_model_performance(model, X, y):
    # Generate predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    sensitivity = recall_score(y, y_pred)  # Also known as recall
    specificity = tn / (tn + fp)
    gmean = np.sqrt(sensitivity * specificity)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"G-Mean: {gmean:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

from sklearn.preprocessing import PolynomialFeatures
def add_polynomial_features_sklearn(df, degree, columns=None):
    """
    Adds polynomial features up to a specified degree to a subset of columns in a Pandas DataFrame using Scikit-Learn's PolynomialFeatures.
    
    Parameters:
        df (Pandas DataFrame): The DataFrame to which the polynomial features will be added.
        degree (int): The maximum degree of polynomial features to add.
        columns (list of str): The names of the columns to which polynomial features will be added. If None, all columns will be used.
        
    Returns:
        Pandas DataFrame: A new DataFrame with the original columns and polynomial features up to the specified degree.
    """
    
    # Select the columns to which polynomial features will be added
    if columns is None:
        columns = df.columns
    df_subset = df[columns]
    
    # Create a copy of the original DataFrame to avoid modifying it
    new_df = df.copy()
    
    # Create a PolynomialFeatures transformer
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Transform the subset of the DataFrame with polynomial features
    poly_df = poly.fit_transform(df_subset)
    
    # Create column names for the new DataFrame
    col_names = [
        f"{col}^d{d}"
        for col in df_subset.columns
        for d in range(1, degree + 1)
    ]
    
    # Create a new DataFrame with the polynomial features
    poly_df = pd.DataFrame(poly_df, columns=col_names, index=df_subset.index)
    
    # Merge the original DataFrame with the new DataFrame
    new_df = pd.concat([new_df, poly_df], axis=1)
    
    return new_df 


def feature_importance(model,X):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    

def analyze_error_distribution(y_true, y_pred):
    """
    Function to analyze the error distribution by plotting histograms and scatter plots.

    Parameters:
    -----------
    y_true : array-like
        Array of true labels or ground truth.
    y_pred : array-like
        Array of predicted values.

    Returns:
    --------
    None
    """
    # Calculate errors
    errors = y_true - y_pred

    # Plot histogram of errors
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=20, alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution (Histogram)')
    plt.grid(True)
    plt.show()

    # Plot scatter plot of true labels vs. errors
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, errors, alpha=0.75)
    plt.xlabel('True Labels')
    plt.ylabel('Error')
    plt.title('Error Distribution (Scatter Plot)')
    plt.grid(True)
    plt.show()

    # Plot scatter plot of predicted values vs. errors
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, errors, alpha=0.75)
    plt.xlabel('Predicted Values')
    plt.ylabel('Error')
    plt.title('Error Distribution (Scatter Plot)')
    plt.grid(True)
    plt.show()
    

def plot_validation_curve(model, X, y, param_name, cv=5, scoring="r2"):
    """
    Plot the validation curve for a given model and hyperparameter.
    
    Parameters:
    - model: Estimator object.
    - X: Input feature matrix.
    - y: Target variable.
    - param_name: Name of the hyperparameter to vary.
    - param_range: Range of hyperparameter values.
    - cv: Number of cross-validation folds (default: 5).
    - scoring: Scoring metric to evaluate (default: "r2").
    """
    # Define the range of hyperparameter values
    param_range = np.arange(1, 21)
    
    # Compute training and validation scores for different hyperparameter values
    train_scores, valid_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring
    )

    # Compute the mean and standard deviation of training and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, label="Training score", color="blue")
    plt.plot(param_range, valid_mean, label="Validation score", color="red")
    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="blue"
    )
    plt.fill_between(
        param_range,
        valid_mean - valid_std,
        valid_mean + valid_std,
        alpha=0.2,
        color="red"
    )
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.title("Validation Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

#the hyperparameter used here is 'max_depth'. A hyperparameter for the model ExtraTreeRegressor


def diagnostic_plots(df, variable):

    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable

    plt.figure(figsize=(15, 6))

    # histogram
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=30)
    plt.title(f"Histogram of {variable}")

    # q-q plot
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.title(f"Q-Q plot of {variable}")

    # check for skewness
    skewness = df[variable].skew()
    if skewness > 0:
        skew_type = "positively skewed"
    elif skewness < 0:
        skew_type = "negatively skewed"
    else:
        skew_type = "approximately symmetric"
        
    # print message indicating skewness type
    print(f"The variable {variable} is {skew_type} (skewness = {skewness:.2f})")
    
    plt.show()


def visualize_outlier (df: pd.DataFrame):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"])
    # Set figure size and create boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    numeric_cols.boxplot(ax=ax, rot=90)
    # Set x-axis label
    ax.set_xlabel("Numeric Columns")
    # Adjust subplot spacing to prevent x-axis labels from being cut off
    plt.subplots_adjust(bottom=0.4) 
    # Increase the size of the plot
    fig.set_size_inches(10, 6)
    # Show the plot
    plt.show()
    


