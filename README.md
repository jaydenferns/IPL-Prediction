# IPL-Prediction
# 🏏 IPL Match Winner Prediction Using Machine Learning

## 📌 Project Overview

This project is a Machine Learning based IPL Match Winner Prediction System developed using historical IPL match data. The system predicts the probable winner of an IPL match using various match-related features such as teams, toss information, venue, and recent team performance.

The project compares multiple machine learning algorithms and implements ensemble learning techniques to improve prediction accuracy.

---

# 🚀 Features

* IPL Match Winner Prediction
* Feature Engineering using Team Form
* Ensemble Learning using Voting Classifier
* Random Forest Classifier
* CatBoost Classifier
* Logistic Regression
* Confusion Matrix Visualization
* Feature Importance Graph
* Classification Report
* Probability-Based Winner Prediction
* Accuracy Comparison Between Models

---

# 🧠 Machine Learning Concepts Used

* Supervised Learning
* Multi-Class Classification
* Ensemble Learning
* Voting Classifier
* Feature Engineering
* OneHotEncoding
* Label Encoding
* Model Evaluation
* Probability Prediction

---

# 📂 Dataset

Dataset used:

```text
IPL_Match_Level.csv
```

The dataset contains IPL match-level historical information such as:

* Batting Team
* Bowling Team
* Toss Winner
* Toss Decision
* Venue
* Match Winner

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib
* Seaborn

---

# 📊 Models Used

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 53%      |
| CatBoost            | 54%      |
| Logistic Regression | 51%      |
| Ensemble Model      | 54%      |

---

# 📈 Visualizations Included

* Confusion Matrix
* Feature Importance Plot
* Accuracy Comparison Graph
* Match Winning Probability Graph

---

# 🏆 Best Performing Model

CatBoost and the Ensemble Model achieved the highest prediction accuracy of approximately:

```text
54%
```

CatBoost performed particularly well because it handles categorical tabular data efficiently.

---

# 🧩 Feature Engineering

The project includes custom engineered features such as:

## Team Form

Recent performance of teams based on:

```text
Last 5 Match Wins
```

This helps the model understand current team momentum.

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/IPL-Match-Winner-Prediction.git
```

---

## Install Required Libraries

```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn
```

---

# ▶️ Run Project

```bash
python main.py
```

---

# 📌 Example Prediction

Input:

```text
Batting Team: Mumbai Indians
Bowling Team: Lucknow Super Giants
Toss Winner: Lucknow Super Giants
Toss Decision: bat
Venue: Ekana Stadium
```

Output:

```text
Predicted Winner: Mumbai Indians
```

---

# 📉 Challenges Faced

* Sports prediction uncertainty
* Limited feature richness
* Class imbalance
* Noisy historical data

---

# 🔮 Future Improvements

Possible future enhancements:

* Player statistics
* Weather conditions
* Pitch reports
* Injury information
* Real-time prediction system
* Web application deployment

---

# 📚 Conclusion

This project successfully demonstrates the application of Machine Learning in sports analytics using IPL historical data. Multiple machine learning models were implemented, compared, and evaluated to build an effective IPL match prediction system.

The project highlights the importance of:

* feature engineering
* ensemble learning
* model evaluation
* categorical data handling

in building practical machine learning systems.

---

# 👨‍💻 Author

Jayden Fernandes

Computer Engineering Student | AIML Enthusiast
