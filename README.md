# AstroPAH_MLDiag
Machine learning pipeline to infer the size and charge of interstellar PAHs from IR spectra.
Author: Zhao Wang @ GXU

This program implements a machine learning pipeline to infer the size and charge of Polycyclic Aromatic Hydrocarbons (PAHs) from their infrared (IR) emission spectra. By transitioning from discrete band ratios to full-spectrum morphology, the model treats the IR profile as a high-dimensional fingerprint to provide probabilistic molecular constraints.

1. Overview

The pipeline uses a Random Forest (RF) Classifier trained on an ensemble of single-molecule spectra. It categorizes molecules into a 12-class framework based on two physical dimensions:

(1) Size: Small (Nc < 50), Medium (50 <= Nc < 100), and Large (Nc >= 100).

(2) Charge: Anion (-1), Neutral (0), Cation (+1), and Dication (+2).

2. Prerequisites

Software Requirements: Python 3.x

- Required Libraries:

numpy, pandas (Data handling)
scikit-learn (Machine Learning)
imbalanced-learn (For SMOTE oversampling)

- Required Data Files:
The following files are necessary for the program to function. Note: These files may be provided in a zipped format to save space; they must be unzipped and placed in the same folder as the Python program before execution.

(1) mol_list.csv: Master metadata containing ID, C (carbon count), and Charge columns.

(2) spectr_bin_all_6eV.txt: Binned training spectra.

(3) spectr_mixed_unseen_6eV.txt: Synthetic mixtures for testing.

(4) unseen_sample_list_6eV.txt: List of molecule IDs reserved strictly for the test set to ensure unbiased evaluation.

3. Core Features

(1) Data Isolation: The program filters out "unseen" molecules from the training set to prevent data leakage.

(2) SMOTE Balancing: To handle class imbalance, the script uses Synthetic Minority Over-sampling Technique (SMOTE) to create a balanced training set.

(3) Mixture Evaluation: Performance is validated on observation-like mixtures with varying population sizes (Nmol).

(4) Interpretability: It exports Feature Importance data, allowing the identification of specific wavelength regions driving the predictions.

4. Usage Instructions

Configuration: Open RF_12class_SMOTE.py and verify the suffix (default is '6eV', can be changed to '3eV' and '9eV') and rf_params match your dataset.

Execution: Run the script via terminal: python RF_12class_SMOTE.py.

Monitoring: The program creates a logs/ directory; check the latest .log file for statistics and the classification report.

5. Output

(1) Logs (/logs): Contains the precision, recall, and F1-score for each of the 12 classes.

(2) Feature Importance (/feature_importance): A CSV file mapping spectral bins to their contribution to the model's decisions.

(3) Confusion Matrix: Printed in the log to show error distributions between classes.

6. Methodology Summary

The model is trained on theoretical (PAHdb) and first-principles (DFT) spectra. 

All spectra are converted to emission profiles using a thermal-cascade approximation assuming an excitation energy of 6 (3 or 9) eV. 

Features are binned onto a grid with a width of 20 cm-1 within the 2.76-20 micron window.

7. License

MIT License - Free to Use

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
