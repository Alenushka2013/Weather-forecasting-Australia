import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report
)

def auroc_train_and_val(model, X_train, X_test, train_targets, test_targets):
    """
    Compute and plot ROC curves for training and validation datasets.
    """

    # Predict probabilities for the positive class
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_test)

    # Compute ROC curves
    fpr_train, tpr_train, _ = roc_curve(train_targets, train_probs[:, 1], pos_label='Yes')
    fpr_val, tpr_val, _ = roc_curve(test_targets, val_probs[:, 1], pos_label='Yes')

    # Compute AUC scores
    auc_train = roc_auc_score(train_targets, train_probs[:, 1])
    auc_val = roc_auc_score(test_targets, val_probs[:, 1])

    # Plot ROC curves
    plt.figure(figsize=(6, 4))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.4f})")
    plt.plot(fpr_val, tpr_val, label=f"Test ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='green')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
def evaluate_classification_model(model, X_train, X_test, train_targets, test_targets):
  train_proba = model.predict_proba(X_train)[:, 1]
  test_proba = model.predict_proba(X_test)[:, 1]

  train_predict  = model.predict(X_train)
  test_predict = model.predict(X_test)

  # Обчислюємо матриці і метрики
  cm_train = confusion_matrix(train_targets, train_predict, normalize='true')
  cm_test = confusion_matrix(test_targets, test_predict, normalize='true')

  # Малюємо поруч
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  ConfusionMatrixDisplay(cm_train).plot(ax=axes[0], colorbar=False)
  axes[0].set_title(f'Train Confusion Matrix')

  ConfusionMatrixDisplay(cm_test).plot(ax=axes[1], colorbar=False)
  axes[1].set_title(f'Test Confusion Matrix')

  plt.tight_layout()
  plt.show()

  # Виведення звіту класифікації для валідаційної вибірки
  print("Звіт класифікації на тестовій вибірці:")
  print(classification_report(test_targets, test_predict, digits=4))
