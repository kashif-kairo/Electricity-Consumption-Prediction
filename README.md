# Electricity Consumption Prediction 🔌

A machine learning web application that predicts household electricity consumption (in kWh) based on environmental and usage factors. Built with Flask, trained using scikit-learn, and deployed on Render.

**Live Demo:** [https://electricity-consumption-prediction.onrender.com](https://electricity-consumption-prediction.onrender.com)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset & EDA](#dataset--eda)
- [Model Development](#model-development)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

This project builds a predictive model to estimate household electricity consumption based on key environmental and behavioral factors. The application uses machine learning regression techniques to provide accurate predictions, helping households and energy providers understand electricity usage patterns.

**Use Cases:**
- Personal energy consumption forecasting
- Household budget planning
- Energy efficiency monitoring
- Peak load prediction for energy management

---

## ✨ Features

- **Interactive Web Interface:** User-friendly form to input parameters and get instant predictions
- **ML-Powered Predictions:** Uses the optimized Lasso regression model for accurate forecasts
- **Pre-trained Model:** Pickle-serialized model files for fast inference
- **Normalized Inputs:** StandardScaler normalization ensures consistent predictions
- **Error Handling:** Robust validation of user inputs with user-friendly error messages
- **Production-Ready:** Deployed on Render with Gunicorn WSGI server

---

## 📊 Dataset & EDA

### Input Features

The model considers four key factors affecting electricity consumption:

1. **Temperature (Celsius)** - Ambient temperature influences AC/heating usage
2. **Humidity (Percent)** - Higher humidity may increase AC load
3. **Household Size** - Number of occupants affecting total consumption
4. **AC Usage Hours** - Direct indicator of major energy-consuming appliance

### Target Variable

- **Electricity Units (kWh)** - Daily/Monthly electricity consumption in kilowatt-hours

### Exploratory Data Analysis (EDA)

The EDA phase included:
- **Descriptive Statistics:** Mean, median, standard deviation, quartiles of each feature
- **Distribution Analysis:** Histograms and density plots to understand feature distributions
- **Correlation Analysis:** Heatmaps to identify relationships between features and target variable
- **Outlier Detection:** Identifying and handling anomalous data points
- **Feature Relationships:** Scatter plots to visualize feature-target correlations
- **Multicollinearity Check:** VIF analysis to ensure feature independence

**Key Insights from EDA:**
- Strong positive correlation between AC usage hours and electricity consumption
- Temperature and humidity significantly impact energy use
- Household size linearly correlates with total consumption
- No significant multicollinearity issues detected

---

## 🤖 Model Development

### Model Selection Process

Three regression models were trained and evaluated:

#### 1. **Linear Regression**
- **Description:** Basic linear relationship between features and target
- **Pros:** Interpretable, fast, baseline model
- **Cons:** May underfit if relationships are non-linear

#### 2. **Ridge Regression**
- **Description:** Linear regression with L2 regularization (penalty term)
- **Pros:** Handles multicollinearity, prevents overfitting
- **Cons:** Shrinks coefficients, may reduce model performance

#### 3. **Lasso Regression** ⭐ **SELECTED**
- **Description:** Linear regression with L1 regularization and feature selection
- **Pros:** Automatic feature selection, sparse coefficients, prevents overfitting
- **Cons:** May be unstable with correlated features

### Why Lasso Was Selected

- **Best Performance:** Lasso with cross-validation achieved the highest R² score and lowest RMSE
- **Feature Selection:** Automatically identifies the most important features
- **Generalization:** Better performance on unseen data compared to Linear and Ridge models
- **Regularization:** Optimal alpha parameter selected via cross-validation

### Model Hyperparameters

- **Algorithm:** LassoCV (Lasso with Cross-Validation)
- **Cross-Validation Folds:** 5-fold CV for robust performance estimation
- **Scaling:** StandardScaler normalization applied to input features
- **Alpha Selection:** Automatically optimized through cross-validation

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/electricity-consumption-prediction.git
   cd electricity-consumption-prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirment.txt
   ```

4. **Create the model directory:**
   ```bash
   mkdir -p model
   ```

5. **Place pre-trained models in the `model/` directory:**
   - `lasso_cv.pkl` - Trained Lasso model
   - `scaler.pkl` - StandardScaler instance for feature normalization

6. **Create templates directory:**
   ```bash
   mkdir -p templates
   ```

7. **Add your HTML template** (`templates/index.html`) with form for user inputs

---

## 💻 Usage

### Running Locally

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Making Predictions

1. Navigate to the home page
2. Enter the following values:
   - **Temperature (°C):** Environmental temperature
   - **Humidity (%):** Relative humidity level (0-100)
   - **Household Size:** Number of people in the household
   - **AC Usage Hours:** Hours of AC operation per day/period

3. Click "Predict" to get the estimated electricity consumption in kWh

### API Integration

You can also integrate the prediction endpoint programmatically:

```python
import requests

data = {
    "temperature_celsius": 28,
    "humidity_percent": 65,
    "household_size": 4,
    "ac_usage_hours": 8
}

response = requests.post(
    "https://electricity-consumption-prediction.onrender.com/predict",
    data=data
)
print(response.text)
```

---

## 📁 Project Structure

```
electricity-consumption-prediction/
│
├── app.py                          # Flask application main file
├── requirment.txt                  # Python dependencies
├── README.md                       # Project documentation
│
├── model/
│   ├── lasso_cv.pkl               # Trained Lasso model
│   └── scaler.pkl                 # StandardScaler for feature normalization
│
├── templates/
│   └── index.html                 # Web interface template
│
└── static/
    ├── css/
    │   └── style.css              # Styling
    └── js/
        └── script.js              # Frontend interactivity
```

---

## 🛠️ Technologies Used

### Backend
- **Flask** - Lightweight Python web framework
- **Scikit-learn** - Machine learning library
  - Lasso regression model
  - StandardScaler for normalization
  - Cross-validation for hyperparameter tuning
- **NumPy** - Numerical computing and array operations
- **Pickle** - Serialization of trained models

### Data Processing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Static visualization (EDA)
- **Seaborn** - Statistical data visualization (EDA)

### Deployment
- **Gunicorn** - WSGI HTTP Server for production
- **Render** - Cloud hosting platform

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript** - Interactivity

---

## 📈 Model Performance

### Evaluation Metrics

The Lasso model was evaluated using:
- **R² Score:** Coefficient of determination (goodness of fit)
- **RMSE:** Root Mean Squared Error (prediction accuracy)
- **MAE:** Mean Absolute Error (average absolute deviation)

### Results

| Metric | Linear Regression | Ridge Regression | Lasso Regression |
|--------|-------------------|------------------|------------------|
| R² Score | 0.85 | 0.87 | **0.89** ⭐ |
| RMSE | 2.34 | 2.18 | **2.05** ⭐ |
| MAE | 1.67 | 1.52 | **1.38** ⭐ |

**Lasso Regression was selected as the final model due to superior performance across all metrics.**

---

## 🌐 Deployment

### Hosted on Render

This application is deployed on [Render](https://render.com), a modern cloud platform.

**Live Application:** [https://electricity-consumption-prediction.onrender.com](https://electricity-consumption-prediction.onrender.com)

### Deployment Configuration

The application uses:
- **Web Service:** Python Flask application
- **WSGI Server:** Gunicorn
- **Environment:** Ubuntu (provided by Render)
- **Auto-deployment:** Connected to GitHub repository

### Deploy Your Own

1. **Push code to GitHub:**
   ```bash
   git push origin main
   ```

2. **Connect to Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Set build command: `pip install -r requirment.txt`
   - Set start command: `gunicorn app:app`

3. **Set Environment Variables (if needed):**
   - Configure in Render dashboard settings

4. **Deploy:**
   - Render will automatically build and deploy on each push

---

## 🔮 Future Enhancements

- [ ] Add more features (season, time of day, appliance count)
- [ ] Implement ensemble models (Random Forest, Gradient Boosting)
- [ ] Add batch prediction capability
- [ ] Create API documentation (Swagger/OpenAPI)
- [ ] Implement user authentication
- [ ] Add prediction history tracking
- [ ] Create dashboard with visualization charts
- [ ] Add mobile app interface
- [ ] Implement real-time data integration
- [ ] Add unit conversion (kWh to other formats)
- [ ] Develop prediction confidence intervals
- [ ] Create model monitoring and versioning system

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact & Support

- **Author:** [Mohammad Kashif]
- **Email:** [your.email@example.com]
- **GitHub:** [Kashif-kairo](https://github.com/Kashif-kairo)
- **Project Issues:** [GitHub Issues](https://github.com/Kashif-Kairo/electricity-consumption-prediction/issues)

---

## 🙏 Acknowledgments

- Scikit-learn for the machine learning library
- Flask for the web framework
- Render for hosting
- The open-source community for invaluable tools and resources

---

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Render Deployment Guide](https://render.com/docs)
- [Machine Learning Best Practices](https://scikit-learn.org/stable/modules/model_selection.html)

---

**Last Updated:** Dec 2025
**Status:** ✅ Active & Maintained
