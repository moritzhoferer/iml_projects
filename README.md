# [Introduction to Machine Learning (252-0220-00L)](http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?semkez=2020S&ansicht=ALLE&lerneinheitId=135514&lang=en)

These projects are part of the Introduction to Machine Learning lecture by Prof. Andreas Krause.
I attended the lecture during the spring semester 2020.
I did the projects only Phd module of the course.
All exercised had to be solved parallel to the lecture.
Below I give shortend descriptions of the tasks.
I used `scikit-learn` and `tensorflow` (v2.1.0).
I do not provide the data files.

Note that the task descriptions are a shortened version of the description given by the lecture and teaching assistants.

## Grades

For the "Projects only" model, I attained an overall 94.71% of the hard baseline and final grade 5.74.

| Task | Public score | Private score | Overall % of hard baseline |
| :------- | :----- | :--- | :--- |
| 1a | 8.356249986          |  n/a   | 100 |
| 1b | 4.909864034          |  n/a     | 100 |
| 2 | 0.74882793           | 0.738429439          | 91.22041912|
| 3 | 0.896210873          | 0.880597015          | 99.89632926 |
| 4 | 0.651509542          | 0.652364831          | 87.73608824 |

## [Task 1a](./task1a.py)

This task is about using cross-validation for ridge regression.
The instructions provide the following set of regularization parameters λ: 0.01, 0.1, 1, 10, and 100.
The task is to perform 10-fold cross-validation with ridge regression for each of the lambdas given above and report the [Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) averaged over the 10 test folds.

## [Task 1b](./task1b.py)

This task is about linear regression: given an input vector **x** (5-dim), our goal is to predict a value **y** as a linear function of a set of feature transformations. Each feature x is transformed in  four ways: linear ϕ(x)=x, quadratic ϕ(x)=x², exponential ϕ(x)=exp(x), consine ϕ (x)=cos(x), where x are the elements of the vector **x**. Additional, we add one constant term. After the transformations, this makes 21 features.

The evaluation metric for this task is the [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation).

## [Task 2](./task2.py)

Patients in hospitals are often continuously monitored by the medical personnel, who collect data about the patients' demographics, vital signs and lab test results.
In this task, we will explore how to forecast future occurrence of medical events such as sepsis, future orders of medical tests, as well as evolution of key vital signs of patients in the remainder of their stay, based on data available from their first 12 recorded hours of stay.

In this task we will face the typical challenges of working with real data: missing features and imbalanced classification, where we are predicting rarely-occuring events.

For evaluation of subtasks 1 and 2, we calculate the Area Under the Receiver Operating Characteristic Curve, aka [ROC AUC scores](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score).
For evaluation of subtask 3, we measure the [R² score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), threshold it below at 0 and normalize it to the range [0.5, 1].
The final score is the average of these 3 scores.

### Subtask 1

We predict whether medical tests are ordered by a clinician in the remainder of the hospital stay: 0 means that there will be no further tests of this kind ordered, 1 means that at least one of a test of that kind will be ordered. For submission, we submit predictions in the interval [0, 1], i.e., the predictions are not restricted to binary.
0.0 indicates we are certain this test will not be ordered, 1.0 indicates we are sure it will be ordered.

### Subtask 2

We predict whether sepsis will occur in the remaining stay: 0 means that no sepsis will occur, 1 otherwise. Similar to Subtask 1, we produce predictions in the interval [0, 1] for this task.

### Subtask 3

We predict future mean values of key vital signs.

## [Task 3](./task3.py)

Proteins are large molecules. Their blueprints are encoded in the DNA of biological organisms. Each protein consists of many amino acids: for example, our protein of interest consists of a little less than 400 amino acids. Once the protein is created (synthesized, it folds into a 3D structure. The mutations influence what amino acids make up the protein, and hence have an effect on its shape.

The goal of this task is to classify mutations of a human antibody protein into active (1) and inactive (0) based on the provided mutation information. Under active mutations the protein retains its original function, and inactive mutation cause the protein to lose its function. The mutations differ from each other by 4 amino acids in 4 respective sites. The sites or locations of the mutations are fixed. The amino acids at the 4 mutation sites are given as 4-letter combinations, where each letter denotes the amino acid at the corresponding mutation site. Amino acids at other places are kept the same and are not provided.

For the evaluation of this task, we use the [F1 score](hhttps://en.wikipedia.org/wiki/F-score) which captures both [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall).

## [Task 4](./task4.py)

In this task, we will make decisions on food taste similarity based on images and human judgement.

The dataset contains images of 10.000 dishes.
Additionally, we provide with a set of triplets (A, B, C) representing human annotations: the human annotator judged that the taste of dish A is more similar to the taste of dish B than to the taste of dish C. A sample of such triplets is shown below.

The task is to predict for unseen triplets (A, B, C) whether dish A is more similar in taste to B or C.
For each triplet (A, B, C), we should predict 0 or 1 as follows:

* 1 if the dish in image A is closer in taste to the dish in image B than to the dish in image C.
* 0 if the dish in image A is closer in taste to the dish in image C than to the dish in image B.

For the evaluation, the accuracy will be measured.
