# 🤖 linear-regression-from-scratch-multivariate
This project implements Multivariate Linear Regression completely from scratch using Python — without relying on machine learning libraries like Scikit-Learn. The goal is to understand the math behind the model, how gradient descent works, and how to build a full ML pipeline manually.


---

## ✨ Features
- Loads and visualizes a multivariate dataset (Size, Bedrooms → Price).
- Implements the hypothesis function for multiple features.
- Computes the Cost Function (MSE) manually.
- Implements Gradient Descent from scratch (vectorized).
- Trains the model to find optimal parameters (θ).
- Evaluates the model using MSE, MAE, and R² score.
- Visualizes the cost function over iterations.
- Saves learned parameters: model_theta.npy.


---


## 📋 Prerequisites
Before running this project, ensure you have:
- Python 3.8+
- NumPy, Pandas, Matplotlib libraries
- CSV dataset data.csv inside a data folder
- Basic knowledge of Python and Linear Regression


---


## 🚀 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/radwanhefny/linear-regression-from-scratch-multivariate.git
cd linear-regression-from-scratch-multivariate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the project:

To run the notebook, simply launch Jupyter Notebook and open the file:
```bash
jupyter notebook linear_regression_from_scratch_multivariate.ipynb
```


---


## 🎬 Screenshots / Demo

### 📉 Cost Function Plot  
Shows how the cost decreases during gradient descent.  
<img src="https://raw.githubusercontent.com/radwanhefny/linear-regression-from-scratch-multivariate/main/pictures/cost%20function.png" width="500"/>


### 🔥 Correlation Heatmap  
Visualizes the relationship between features and the target variable.  
<img src="https://raw.githubusercontent.com/radwanhefny/linear-regression-from-scratch-multivariate/main/pictures/correlation-heatmap.png" width="500"/>


### 📊 Scatter Plot: Size vs Price  
Shows how house size affects price.  
<img src="https://raw.githubusercontent.com/radwanhefny/linear-regression-from-scratch-multivariate/main/pictures/size-price-scatter.png" width="500"/>


### 🛏️ Scatter Plot: Bedrooms vs Price  
Shows the relationship between number of bedrooms and house price.  
<img src="https://raw.githubusercontent.com/radwanhefny/linear-regression-from-scratch-multivariate/main/pictures/bedrooms-price-scatter.png" width="500"/>



---



## 🗂️ Project Structure
```
📁 linear-regression-from-scratch-multivariate
├── linear_regression_from_scratch_multivariate.ipynb   # Core logic: cost, gradient descent, training
├── data/
│   └── data.csv              # Dataset: size, bedrooms, price
├── results/
│   ├── error.png             # Cost function over iterations
│   └── scatter.png           # Scatter plots of features vs price
├── model_theta.npy           # Saved learned parameters
├── requirements.txt
└── README.md
```


---


## 🛠️ Usage
Run the notebook to train the model and generate results.
Expected output:
- error.png → Cost function vs iterations
- model_theta.npy → Saved parameters
- scatter.png → Feature visualization
Expected performance (approximate):
- R² Score: ~0.70–0.75
- MSE & MAE depend on dataset scale (raw housing prices)


---


## ✅ Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² score


---


## 🧠 How It Works
1. Loads the dataset using Pandas.
2. Separates X (size, bedrooms) and y (price).
3. Normalizes features manually or using standardization.
4. Adds a column of ones for the bias term.
5. Implements the hypothesis function.
6. Implements Cost Function.
7. Implements Gradient Descent (vectorized).
8. Updates parameters until convergence.
9. Plots the cost function to visualize learning progress.


---


## 🔗 Related Repositories

- 📊 **Optimization Dashboard**  
  Visualizes Gradient Descent behavior step by step  
  https://github.com/radwanhefny/Gradient-Descent-Optimization-Dashboard



---


## 🤝 Contributing
Contributions are welcome!
1. Fork the repository
2. Create a new feature branch
3. Submit a pull request
Please ensure your code is clean, structured, and well-commented.


---


## 📝 License
This project is licensed under the MIT license - see the LICENSE file for details. 


---


## 📞 Support
If you have questions or need help, feel free to:
- Open an issue on this repository  
- Connect with me on LinkedIn: https://www.linkedin.com/in/radwanhefny  
