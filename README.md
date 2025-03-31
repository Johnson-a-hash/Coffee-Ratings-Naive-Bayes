# Predicting Coffee Ratings Using Naive Bayes

## Overview
This project implements a Naive Bayes classification model from scratch to predict coffee ratings based on attributes such as roaster, roast level, origin, price, and textual review data. The model classifies coffee as either average (class 0) or outstanding (class 1).

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Evaluation & Results](#evaluation--results)
- [Files in this Repository](#files-in-this-repository)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Challenges](#challenges)
- [References](#references)

## Methodology
The Naive Bayes algorithm operates under the assumption of feature independence and applies Bayes’ Theorem with strong (naive) independence assumptions between features. The steps followed include:

1. **Preprocessing Textual Data**: Tokenization, stopword removal, and lemmatization using NLTK.
2. **Feature Extraction**: Using TF-IDF vectorization to convert text into numerical features.
3. **Feature Selection**: Applying SelectKBest with ANOVA F-value to select the most relevant features.
4. **Model Implementation**: Training a Multinomial Naive Bayes classifier using scikit-learn.
5. **Evaluation**: Measuring performance using accuracy and F1-score through cross-validation.

## Evaluation & Results
The final evaluation of the model produced:
- **Accuracy**: 95.97%
- **F1-Score**: 95.39%

The optimal number of attributes selected was 161. This model achieved high validation scores with minimal discrepancies between training and validation scores, demonstrating generalizability and robustness.

## Files in this Repository
- `Predicting_Coffee_Ratings_Using_Naive_Bayes_AJ.pdf`: Project report detailing methodology, implementation, results, and conclusions.
- `Code.ipynb`: Python notebook containing the implementation of the Naive Bayes model.
- `format.csv`: File used for formatting data.
- `result.csv`: File containing results from the model.
- `X_test.csv`, `X_train.csv`, `y_train.csv`: Training and testing datasets.
- `README.md`: This file.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib (for visualizations)

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
```
2. Navigate to the project directory:
```bash
cd Predicting_Coffee_Ratings_Using_Naive_Bayes
```
3. Install the required dependencies.
4. Run the Jupyter Notebook `Code.ipynb` to reproduce the results.

## Challenges
- Processing unstructured textual data for feature extraction.
- Selecting features that best differentiate between the two classes.
- Visualizing the decision-making process effectively.

## References
- Bird, Steven, et al. *Natural Language Processing with Python.* O'Reilly Media, Inc., 2009.
- Manning, Christopher D., et al. *Introduction to Information Retrieval.* Cambridge University Press, 2008.
- McCallum, Andrew, and Kamal Nigam. *A comparison of event models for naive bayes text classification.* AAAI-98 workshop, 1998.
- Pedregosa, Fabian, et al. *Scikit-learn: Machine learning in Python.* Journal of machine learning research, 2011.
- Powers, David MW. *Evaluation: from precision, recall and f-measure to roc, informedness, markedness and correlation.* Journal of Machine Learning Technologies, 2011.
- Rennie, Jason D M, et al. *Tackling the poor assumptions of naive bayes text classifiers.* International Conference on Machine Learning (ICML-03), 2003.
