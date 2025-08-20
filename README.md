
# ❤️ Heart Disease Prediction Project

This project predicts the presence of **heart disease** in patients using **machine learning models**.
We apply **data preprocessing, feature engineering, dimensionality reduction, and multiple classification algorithms**, then evaluate and compare their performance.

---

## 🚀 Deployment

The trained Logistic Regression model has been deployed as an interactive **Streamlit web app**.

👉 **Access the App here:**
[HeartRisk-AI Web App](https://heartrisk-ai.streamlit.app/)

### How it works:

1. Enter patient medical parameters (age, cholesterol, blood pressure, etc.).
2. The model processes the inputs (scaling + PCA).
3. Predicts whether the patient is at **risk of heart disease** (`1`) or **not at risk** (`0`).
4. Displays prediction with probability/accuracy.

---

## 📊 Dataset

* **Source**: Heart Disease Statlog Dataset
* **Samples**: 270 patients
* **Features**:

  * **Continuous**: age, trestbps (resting blood pressure), chol (cholesterol), thalach (max heart rate), oldpeak (ST depression), etc.
  * **Categorical**: sex, cp (chest pain type), fbs (fasting blood sugar), restecg, exang (exercise induced angina), slope, ca, thal
* **Target**:

  * `0` → No Heart Disease
  * `1` → Heart Disease

---
##  Images and deploymentlink
<img width="1693" height="826" alt="image" src="https://github.com/user-attachments/assets/3063c66e-9007-47ac-aad9-69f5623726b4" />

<img width="1846" height="866" alt="image" src="https://github.com/user-attachments/assets/4945d121-d755-4a7f-95a4-cf2d69c85b07" />

<img width="1854" height="872" alt="image" src="https://github.com/user-attachments/assets/212b8aa5-aa66-479b-9cc4-0af5a93829b3" />

---

## ⚙️ Data Preprocessing

1. **Feature Separation**

   * Continuous features processed with correlation analysis.
   * Categorical features processed with Cramér’s V correlation.

2. **Feature Selection**

   * Removed features with **high inter-correlation**.
   * Removed statistically insignificant ones (ANOVA & Chi-squared).

3. **Scaling & Dimensionality Reduction**

   * Continuous features standardized using `StandardScaler`.
   * PCA applied → Retained 95% variance.
   * MCA optional for categorical features.

4. **Outlier Handling**

   * Outliers checked with boxplots & IQR.

---

## 📊 Exploratory Data Analysis

* **Class Distribution**:
  Balanced between patients with and without disease.

* **Visualizations**:

  * Heatmaps of correlations.
  * Histograms & KDE plots for continuous features.
  * Boxplots for outlier detection.
  * Countplots for categorical features.

---

## 🤖 Machine Learning Models

The following models were trained, tuned, and compared:

| Model                        | Accuracy                 | Notes                                                         |
| ---------------------------- | ------------------------ | ------------------------------------------------------------- |
| **Logistic Regression**      | **92.59%**               | High accuracy, low false positives → Best medical choice      |
| **SVM (RBF Kernel)**         | **92.59%**               | Same accuracy as Logistic Regression but more false positives |
| **Naive Bayes (GaussianNB)** | 90.74%                   | Simple, but slightly lower accuracy                           |
| **Random Forest**            | \~88%                    | Good generalization, but underperformed here                  |
| **KNN**                      | \~85–89% (varies with k) | Sensitive to `k` and scaling                                  |
| **Decision Tree**            | \~80–85%                 | Prone to overfitting, less stable                             |

---

## 📉 Confusion Matrices

### Logistic Regression (92.59%)

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 27          | 3           |
| **Actual 1** | 1           | 23          |

* Few false positives → **safest for medical use**.

---

### SVM (92.59%)

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 26          | 4           |
| **Actual 1** | 1           | 23          |

* Slightly more false positives than Logistic Regression.

---

### Naive Bayes (90.74%)

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 25          | 5           |
| **Actual 1** | 2           | 22          |

* More misclassifications compared to Logistic Regression.

---

### Random Forest (\~88%)

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 24          | 6           |
| **Actual 1** | 3           | 21          |

* Weaker performance compared to linear models.

---

### KNN (Best \~88%)

* Performance fluctuates depending on **k**.
* Best accuracy at `k = 7–9`, around **88%**.

---

### Decision Tree (\~82%)

* Accuracy lower compared to other models.
* Highly sensitive to `max_depth`.

---
<img width="1608" height="738" alt="image" src="https://github.com/user-attachments/assets/57fd5c9f-1c4b-49f0-87cb-6f52f5630787" />

---

## 📊 Model Accuracy Comparison

![Model Accuracy Comparison](comparison_plot.png)

* **Best Models**: Logistic Regression & SVM (92.59%)
* **Selected Final Model**: **Logistic Regression** (because fewer false positives).

---

## 🏆 Final Model

* Final model: **Logistic Regression**
* Saved as `finalized_model.pkl`
* Pickle includes:

  * Accuracy score
  * Model object
  * Scaler
  * PCA object

---

## ▶️ How to Run?

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/HeartDiseaseProject.git
   cd HeartDiseaseProject
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook HeartDiseaseProject.ipynb
   ```

4. Load trained model for prediction:

   ```python
   import pickle
   with open("finalized_model.pkl", "rb") as file:
       accuracy = pickle.load(file)
       model = pickle.load(file)
       scaler = pickle.load(file)
       pca = pickle.load(file)

   print("Model Accuracy:", accuracy)
   ```

---

## 📌 Future Improvements

* Deployment with **Streamlit / Flask** for real-time predictions.
* Try **Boosting models** (AdaBoost, XGBoost, Gradient Boosting).
* Hyperparameter tuning with **GridSearchCV**.
* Use larger, more diverse datasets for generalization.

---

✅ **Final Choice:** Logistic Regression with **92.59% accuracy** and **lowest false positive rate** → making it most suitable for **medical applications**.


