
# Deep Neural Network for Wine Classification

## 🧾 Overview

This project addresses the problem of **classifying wine types (red or white) based on their chemical properties using a deep learning approach.** The dataset utilized consists of various chemical measurements of red and white wines sourced from a publicly available **UCI Machine Learning Repository.** The modeling status involves training and evaluating a **Sequential neural network** for **binary classification.**

---

## 🧠 Business Understanding

While the code itself doesn't explicitly define a specific business stakeholder, a potential business problem could be related to quality control, authentication, or automated sorting in the wine industry. For example, a winery or distributor might need to quickly and accurately identify the type of wine based on its composition, perhaps for inventory management, preventing mislabeling, or detecting counterfeit products. Another application could be assisting sommeliers or consumers by providing data-driven insights into wine characteristics.


- E-commerce (product tagging and recommendations)
- Healthcare (diagnostic imaging)
- Manufacturing (defect detection)

This project simulates how a deep learning pipeline can be applied to a structured visual dataset to automate classification tasks efficiently.

---

## 📊 Data Understanding

The data used in this project is a combination of two datasets from the UCI Machine Learning Repository: red wine quality and white wine quality. The **data consists of 12 chemical features** and a **target variable indicating the wine type (red or white).** The combined **dataset contains 6497 samples.** The timeframe of the data is not explicitly stated in the provided code. A limitation of this dataset might be that it's from a specific source (P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, 47(4):547-553, 2009.), and the results might not generalize perfectly to all wines from all regions and vintages. The provided visualizations include a distribution plot of alcohol content for all wines and separate histograms for red and white wines, revealing differences in their alcohol distributions.

![image](https://github.com/user-attachments/assets/30c176ec-5ec0-4b78-94bf-7f594c6779fc)


Dataset used: **UCI Wine Dataset** 

- 6497 samples + 12 chemical features

---

## 🤖 Modelling & Evaluation

A **Sequential deep learning model** was used for binary classification. The model consists of **an input layer** (implicitly defined by the first Dense layer's input shape), **two hidden Dense layers with ReLU activation (16 and 8 neurons respectively)**, and **an output Dense layer with a sigmoid activation for binary classification.** The model was **compiled using the 'binary_crossentropy' loss function**, the **'adam' optimizer**, and **'accuracy' as the evaluation metric**. The model was trained for **3 epochs with a batch size of 1.**

### Deep Neural Network Architecture:
- Input Layer:  12 features
- Hidden Layers: Dense layers with ReLU activation
- Output Layer: 1 units (Sigmoid for binary classification)

### Training:
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy

### Evaluation:
- Training vs validation loss and accuracy visualization
- Confusion Matrix
- Classification Report with precision, recall, F1-score

  ![image](https://github.com/user-attachments/assets/1860d8ce-cb05-44d2-b0d7-dd8f29bafe2b)


---

## 📌 Conclusion

The model performed well in classifying wine types, achieving an overall **accuracy of 95.15%**. The classification report shows high **precision and recall for white wine (0.95 and 0.98 respectively)**, indicating that the model is very good at identifying white wines and **minimizing false positives and negatives for this class.** For **red wine**, the **precision is also high (0.95), but the recall is slightly lower (0.86)**, meaning the model is slightly less effective at identifying all actual red wines. The **AUC of 0.92 further confirms the model's strong ability to distinguish between the two wine types.**

Based on these results, a recommendation for solving a business problem like quality control could be to deploy this model to automatically classify wines based on their chemical analyses. This could speed up processes and reduce human error. In the future we can perform hyperparameter tunning, address class imbalance, perform cross validation, data augmentation to improve the oerformance of the model.

---

## ✅ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn (for evaluation)

---
