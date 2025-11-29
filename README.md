# 📊 linear-regression-from-scratch-multivariate
This project implements Multivariate Linear Regression completely from scratch using Python — without relying on machine learning libraries like Scikit-Learn. The goal is to understand the math behind the model, how gradient descent works, and how to build a full ML pipeline manually.

## ✨ Features
- Loads and visualizes the dataset (Population vs Profit).
- Implements the hypothesis function manually.
- Computes Cost Function (MSE) step by step.
- Implements Gradient Descent without any ML libraries.
- Trains the model to find optimal parameters (θ).
- Evaluates the model using MSE, MAE, and R².
- Visualizes the predicted regression line and error over iterations.
- Saves learned parameters (model_theta.npy) for reuse.
## 📋 Prerequisites
Before running this project, ensure you have:
- Python 3.8+
- NumPy, Pandas, Matplotlib libraries
- CSV dataset data.csv inside a data folder
- Basic knowledge of Python and Linear Regression
## 🚀 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/your-username/linear-regression-from-scratch-univariate.git
cd linear-regression-from-scratch-univariate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the project:
```bash
python linear_regression_from_scratch_univariate.py
```
## 🎬 Screenshots / Demo

### 📈 Prediction Line Plot
![Prediction Figure](https://github.com/radwanhefny/linear-regression-from-scratch-univariate/blob/main/results/prediction.png)


---

### 📉 Error Plot
![Error Figure](https://github.com/radwanhefny/linear-regression-from-scratch-univariate/blob/main/results/error.png)


## 🗂️ Project Structure
```
📁 linear-regression-from-scratch-univariate
├── linear-regression-from-scratch-univariate.py   # Core calculations: cost, gradient descent, training
├── data/
│   └── data.csv           # Dataset(The data contains one independent variable, which is population size, │                                     and one dependent variable, which is profit.)
├── results/
│   ├── prediction.png     # Predicted regression line
│   └── error.png          # Error over iterations
├── model_theta.npy        # Saved model parameters for reuse
├── requirements.txt
└── README.md



```
## 🛠️ Usage
Run the script to see results.

Expected output:
- prediction.png → Predicted line vs training data
- error.png → Cost function over training iterations
- Expected R^2 Score: ~0.70 (This score is considered very good, given that the model is univariate and working on real/experimental data.)

## ✅ Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² score
## 🧠 How It Works
1. Reads the dataset using Pandas.
2. Adds a column of ones for the bias term.
3. Separates X (features) and y (target).
4. Implements Cost Function (MSE) manually.
5. Implements Gradient Descent step by step to update θ.
6. Trains the model to minimize the cost.
7. Visualizes the regression line and training error over iterations.

## 🤝 Contributing
Contributions are welcome!
1. Fork the repository
2. Create a new feature branch
3. Submit a pull request
Please ensure your code is clean, structured, and well-commented.
## 📝 License
This project is licensed under the MIT license - see the LICENSE file for details. 
## 📞 Support
If you have questions or need help, feel free to:
- Open an issue on this repository  
- Connect with me on LinkedIn: https://www.linkedin.com/in/radwanhefny  
