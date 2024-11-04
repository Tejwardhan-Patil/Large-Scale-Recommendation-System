import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model, test_data, true_labels):
        self.model = model
        self.test_data = test_data
        self.true_labels = true_labels
        self.predictions = None
        self.binary_predictions = None

    def predict(self):
        """Make predictions on the test data using the model."""
        self.predictions = self.model.predict(self.test_data)
        return self.predictions

    def calculate_rmse(self):
        """Calculate Root Mean Squared Error (RMSE) between predictions and true labels."""
        if self.predictions is None:
            self.predict()
        return np.sqrt(mean_squared_error(self.true_labels, self.predictions))

    def calculate_r2_score(self):
        """Calculate R-squared (Coefficient of Determination) score."""
        if self.predictions is None:
            self.predict()
        return r2_score(self.true_labels, self.predictions)

    def calculate_explained_variance(self):
        """Calculate the explained variance score."""
        if self.predictions is None:
            self.predict()
        return explained_variance_score(self.true_labels, self.predictions)

    def calculate_binary_predictions(self, threshold=0.5):
        """Convert predictions to binary format based on a given threshold."""
        if self.predictions is None:
            self.predict()
        self.binary_predictions = (self.predictions >= threshold).astype(int)
        return self.binary_predictions

    def calculate_accuracy(self):
        """Calculate accuracy for classification models."""
        if self.binary_predictions is None:
            self.calculate_binary_predictions()
        return accuracy_score(self.true_labels, self.binary_predictions)

    def calculate_precision(self):
        """Calculate precision score."""
        if self.binary_predictions is None:
            self.calculate_binary_predictions()
        return precision_score(self.true_labels, self.binary_predictions)

    def calculate_recall(self):
        """Calculate recall score."""
        if self.binary_predictions is None:
            self.calculate_binary_predictions()
        return recall_score(self.true_labels, self.binary_predictions)

    def calculate_f1(self):
        """Calculate F1 score."""
        if self.binary_predictions is None:
            self.calculate_binary_predictions()
        return f1_score(self.true_labels, self.binary_predictions)

    def calculate_roc_auc(self):
        """Calculate Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
        if self.predictions is None:
            self.predict()
        return roc_auc_score(self.true_labels, self.predictions)

    def generate_confusion_matrix(self):
        """Generate and display the confusion matrix."""
        if self.binary_predictions is None:
            self.calculate_binary_predictions()
        cm = confusion_matrix(self.true_labels, self.binary_predictions)
        self.plot_confusion_matrix(cm)
        return cm

    def plot_confusion_matrix(self, cm):
        """Plot the confusion matrix using seaborn heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_residuals(self):
        """Plot the residuals (difference between true values and predictions) for regression models."""
        if self.predictions is None:
            self.predict()
        residuals = self.true_labels - self.predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(self.predictions, residuals, alpha=0.6, color='blue')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    def plot_prediction_distribution(self):
        """Plot the distribution of the predicted values."""
        if self.predictions is None:
            self.predict()
        plt.figure(figsize=(10, 6))
        sns.histplot(self.predictions, bins=50, kde=True, color='green')
        plt.title('Prediction Distribution')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.show()

    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report."""
        report = {}
        
        if self.predictions is None:
            self.predict()

        # Regression metrics
        report['RMSE'] = self.calculate_rmse()
        report['R-squared'] = self.calculate_r2_score()
        report['Explained Variance'] = self.calculate_explained_variance()

        # Classification metrics
        if len(np.unique(self.true_labels)) == 2:
            report['Accuracy'] = self.calculate_accuracy()
            report['Precision'] = self.calculate_precision()
            report['Recall'] = self.calculate_recall()
            report['F1 Score'] = self.calculate_f1()
            report['ROC AUC'] = self.calculate_roc_auc()
            report['Confusion Matrix'] = self.generate_confusion_matrix()

        return report

    def evaluate(self):
        """Main evaluation function to assess model performance."""
        evaluation_report = self.generate_evaluation_report()

        print("\nEvaluation Report:")
        for metric, value in evaluation_report.items():
            if metric == 'Confusion Matrix':
                print(f"{metric}:")
                print(value)
            else:
                print(f"{metric}: {value:.4f}")

        if len(np.unique(self.true_labels)) > 2:  # Regression plot
            self.plot_residuals()
        else:  # Classification plot
            self.plot_prediction_distribution()

    def save_report(self, file_path="evaluation_report.csv"):
        """Save the evaluation report to a CSV file."""
        report = self.generate_evaluation_report()

        # Flatten the confusion matrix into a simple format if it exists
        if 'Confusion Matrix' in report:
            cm = report.pop('Confusion Matrix')
            report['True Negative'], report['False Positive'], report['False Negative'], report['True Positive'] = cm.ravel()

        df = pd.DataFrame([report])
        df.to_csv(file_path, index=False)
        print(f"Evaluation report saved to {file_path}")

    def plot_roc_curve(self):
        """Plot the ROC curve for binary classification models."""
        if len(np.unique(self.true_labels)) != 2:
            print("ROC curve is only applicable for binary classification tasks.")
            return

        if self.predictions is None:
            self.predict()

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.predictions)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {self.calculate_roc_auc():.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def feature_importance(self):
        """Plot feature importance for models that support it."""
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance = self.model.feature_importances_
            feature_names = self.test_data.columns
            sns.barplot(x=importance, y=feature_names, color='teal')
            plt.title('Feature Importance')
            plt.show()
        else:
            print("The model does not support feature importance.")