"""

Notes:

The script handles variable numbers of PRS and RRS probability values.
It combines the probabilities and labels, computes the ROC curve, calculates the AUC (Area Under the Curve), and plots the ROC curve using Matplotlib.
You can customize the script to read multiple PRS and RRS files by modifying the read_probabilities function and how you handle the input arguments.
Dependencies:

Python 3
numpy
matplotlib
scikit-learn

Explanation of the Area Under the Curve (AUC):

The Area Under the Curve (AUC) refers to the area under the Receiver Operating Characteristic (ROC) curve. The ROC curve is a graphical representation of a classifier's performance across all classification thresholds. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

True Positive Rate (TPR), also known as Sensitivity or Recall, is the proportion of actual positives that are correctly identified.
False Positive Rate (FPR) is the proportion of actual negatives that are incorrectly identified as positives.
The AUC provides a single scalar value that summarizes the performance of the classifier:

An AUC of 1.0 indicates a perfect classifier.
An AUC of 0.5 suggests no discriminative ability (equivalent to random guessing).
An AUC between 0.5 and 1.0 indicates the degree to which the classifier can distinguish between the positive and negative classes.
Why is AUC important?

More on interpreting the ROC Curve:

The ROC curve plots the TPR against the FPR at various threshold levels.
The closer the curve follows the left-hand border and then the top border of the ROC space, the better the classifier.
The diagonal line represents the performance of a classifier that makes random guesses.
Understanding AUC Values:

AUC = 0.90-1.00: Excellent
AUC = 0.80-0.90: Good
AUC = 0.70-0.80: Fair
AUC = 0.60-0.70: Poor
AUC = 0.50-0.60: Fail



Threshold-Independent: AUC measures the classifier's ability to rank predictions without being dependent on a specific threshold.
Performance Metric: It provides a comprehensive measure of performance across all possible classification thresholds.
In summary, the AUC quantifies the overall ability of the model to discriminate between positive and negative classes. A higher AUC indicates better model performance.



F1 = 2 * ( (precision * recall) / (precision + recall) )

precision = TP / (TP + FP)

recall = TP / (TP + FN)


You can adjust the decimal percision by changing ".6f" to desired value in  f'Best F1 Threshold: {best_thresh:.6f}'


"""
#

# pip install numpy matplotlib scikit-learn

# python roc_analysis_color_threshold_F1e.py --input_csv probabilities.csv --output_file roc_curve.png

#!/usr/bin/env python

#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import csv

def read_probabilities_from_csv(filename):
    """Read PRS and RRS probability values from a CSV file."""
    prs_probs = []
    rrs_probs = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header row
        for row in reader:
            # Ensure there are at least two columns
            if len(row) >= 2:
                prs_value = row[0].strip()
                rrs_value = row[1].strip()
                # Append PRS probability if not empty
                if prs_value:
                    prs_probs.append(float(prs_value))
                # Append RRS probability if not empty
                if rrs_value:
                    rrs_probs.append(float(rrs_value))
    return prs_probs, rrs_probs

def main():
    parser = argparse.ArgumentParser(description='Compute ROC curve, best F1 score, and annotate thresholds.')
    parser.add_argument('--input_csv', required=True, help='CSV file containing PRS and RRS probability values')
    parser.add_argument('--output_file', default='roc_curve.png', help='Output file name for ROC curve plot')

    args = parser.parse_args()

    # Read probability values from CSV file
    prs_probs, rrs_probs = read_probabilities_from_csv(args.input_csv)

    # Assign labels
    prs_labels = [1] * len(prs_probs)
    rrs_labels = [0] * len(rrs_probs)

    # Combine probabilities and labels
    probs = np.array(prs_probs + rrs_probs)
    labels = np.array(prs_labels + rrs_labels)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Exclude infinite thresholds and thresholds outside [0, 1]
    finite_idxs = np.where(np.isfinite(thresholds))[0]
    fpr = fpr[finite_idxs]
    tpr = tpr[finite_idxs]
    thresholds = thresholds[finite_idxs]

    # Filter thresholds within [0, 1]
    valid_thresholds_idxs = np.where((thresholds >= 0) & (thresholds <= 1))[0]
    fpr = fpr[valid_thresholds_idxs]
    tpr = tpr[valid_thresholds_idxs]
    thresholds = thresholds[valid_thresholds_idxs]

    # Compute best F1 score across thresholds
    best_f1 = -1.0
    best_thresh = None
    best_idx = None
    for i, thresh in enumerate(thresholds):
        predicted_labels = (probs >= thresh).astype(int)
        current_f1 = f1_score(labels, predicted_labels)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh
            best_idx = i

    # Retrieve FPR and TPR for the best threshold
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    # Set global font
    plt.rcParams['font.family'] = 'Arial'

    # Create figure and colormap
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = plt.Normalize(vmin=thresholds.min(), vmax=thresholds.max())
    cmap = plt.cm.viridis

    # Plot the ROC curve in segments, color-coded by threshold
    for i in range(len(fpr) - 1):
        x = fpr[i:i + 2]
        y = tpr[i:i + 2]
        z = thresholds[i]
        ax.plot(x, y, color=cmap(norm(z)), lw=2.5)

    # Diagonal line
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    # Add a colorbar for thresholds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Threshold', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Annotate a subset of thresholds on the ROC curve
    num_thresholds_to_annotate = 10  # Number of thresholds to annotate
    idxs = np.linspace(0, len(thresholds) - 1, num_thresholds_to_annotate).astype(int)
    for idx in idxs:
        thresh = thresholds[idx]
        ax.annotate(f'{thresh:.2f}', xy=(fpr[idx], tpr[idx]),
                     textcoords='offset points', xytext=(0, 10),
                     ha='center', fontsize=12, color='blue')

    # No red scatter point for the best threshold
    # ax.scatter(best_fpr, best_tpr, color='red', s=100, zorder=5)

    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend with 3 decimal places
    legend_text = (f'ROC curve (AUC = {roc_auc:.3f}, '
                   f'Best F1 = {best_f1:.3f}, '
                   f'Best F1 Threshold = {best_thresh:.3f})')
    ax.legend([legend_text], loc="lower right", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save and show the figure
    plt.savefig(args.output_file, dpi=300, format='png')
    plt.show()

    print(f"ROC curve saved to {args.output_file}")
    print(f"Best F1 Score: {best_f1:.3f} at threshold {best_thresh:.3f}")

if __name__ == '__main__':
    main()
