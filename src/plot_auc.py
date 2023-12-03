import json
import matplotlib.pyplot as plt
import os

import os

def plot_roc_from_json(json_file_path, output_folder='roc_plots'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(json_file_path, 'r') as file:
        roc_auc_entries = json.load(file)

    for idx, data in enumerate(roc_auc_entries):
        plt.figure()
        plt.plot(data['fpr'], data['tpr'], label=f"ROC Curve (area = {data['roc_auc']:.2f}) - Run {idx + 1}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curves")
        plt.legend(loc="lower right")

        # Save the plot as an image file
        plt.savefig(os.path.join(output_folder, f'roc_curve_run_{idx + 1}.png'))
        plt.close()


if __name__ == "__main__":
    json_file_path = 'plotData\\lstm50unit_plot.json'  # Update this path to your JSON file
    plot_roc_from_json(json_file_path)
