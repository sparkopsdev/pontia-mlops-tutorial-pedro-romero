import logging
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger("adult-income")


def evaluate(model, X_test, y_test):
    logger.info("Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    logger.info("Test Accuracy: %.4f", acc)
    logger.info("Classification Report:\n%s", report)
    return acc, report
