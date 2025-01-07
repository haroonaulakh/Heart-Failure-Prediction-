🫀 Heart Failure Prediction
An interactive web application for analyzing clinical data, predicting heart failure risks, and gaining actionable insights to improve healthcare outcomes.

🌟 Features
Data Exploration:

View descriptive statistics and filter datasets based on parameters like age and gender.
Interactive visualizations using Plotly and Seaborn.
Survival Analysis:

Kaplan-Meier survival curves to estimate survival probabilities over time.
Stratified analysis by demographic factors like gender.
Correlation Analysis:

Heatmaps and scatterplots to uncover relationships between clinical parameters.
Highlight influential features such as Ejection Fraction and Serum Creatinine.
Predictive Modeling:

Logistic Regression model with an accuracy of 87%.
Real-time predictions for new patient data, including probabilities for survival or adverse outcomes.
🚀 Getting Started
Prerequisites
Ensure you have the following installed:

Python 3.8 or later
Required Python libraries (listed in requirements.txt)
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/heart-failure-prediction.git
cd heart-failure-prediction
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Application:

bash
Copy code
streamlit run Heart_Failure_Prediction.py
Access the app in your browser at http://localhost:8501.

🧪 Dataset
The clinical data used for this project is sourced from Kaggle Heart Failure Dataset.

Key Features in the Dataset:
Age: Patient's age.
Ejection Fraction: Percentage of blood leaving the heart during contraction.
Serum Creatinine: Indicator of kidney function.
DEATH_EVENT: Binary indicator of survival (1 = death, 0 = survival).
📊 Machine Learning Model
The Logistic Regression model predicts heart failure risks with high accuracy.

Target Variable: DEATH_EVENT.
Performance Metrics:
Accuracy: 87%.
Confusion matrix and classification report available in the app.
📈 Visualizations
The application uses Plotly for interactive charts and Seaborn/Matplotlib for static visualizations. Some highlights:

Survival Curves: Kaplan-Meier estimates for overall and group-specific survival probabilities.
Feature Correlation Heatmap: Relationships between numerical features and the target variable.
💡 Future Enhancements
Add support for additional machine learning models (e.g., Random Forest, SVM).
Implement SHAP plots for feature importance analysis.
Enhance survival analysis with stratifications based on additional clinical features.
👩‍💻 Technologies Used
Frontend: Streamlit
Backend: Python
Machine Learning: Scikit-learn
Data Visualization: Plotly, Seaborn, Matplotlib
🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request with your enhancements.

📜 License
This project is licensed under the MIT License.

🔗 Links
Kaggle Dataset: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
GitHub Repository: https://github.com/haroonaulakh/Heart-Failure-Prediction-.git
