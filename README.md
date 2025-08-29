# Investigation of the effect of the Lambda regularization parameter (λ)

This project demonstrates the manual implementation of linear regression with L2 regularization (Ridge Regression) and compares it with the `LinearRegression' model from the `scikit-learn' library. The main attention is paid to the study of the effect of the regularization hyperparameter `lambda` (λ) on the quality of the model.

## Description

Regularization is used to prevent overfitting of the model. The `lambda' parameter controls the strength of regularization:
- **lambda = 0**: Corresponds to conventional linear regression (OLS) without regularization.
- **lambda > 0**: L2 regularization is applied. The higher the value of `lambda`, the stronger the regularization, which can lead to a decrease in the variance of the model by increasing the bias.

## Implementation

- **Manual implementation**: The Ridge regression model is implemented analytically using the formula:
  `w = (X^T X + λI)^{-1} X^T y`
- bias/intercept is taken into account by adding a column of units to the feature matrix `X`.
  - Regularization is applied only to feature weights, not to bias.
- **Comparison**: The results are compared with the `sklearn.linear_model.LinearRegression` model (which does not use regularization by default).
- **Quality Metric**: RMSE (Root Mean Square Error) is used on the test sample.
- **Visualization**: A graph of the dependence of RMSE on the training and test samples on the value of `lambda' is constructed.

## Results

The RMSE vs. Lambda graph shows typical U-shaped behavior for a test error:
- With **small** values of `lambda', the model can **overfit** (low bias, high variance), which leads to a high error on the test data.
- With **large** values of `lambda`, the model becomes too simple (**under-learning**), also increasing the error.
- There is an optimal lambda value at which the test error is minimal, providing the best balance between bias and variance.

## Launch

1. Make sure that the necessary libraries are installed:
``bash
pip install numpy pandas scikit-learn matplotlib
    ```
2. Run the basic Python script (for example, `ridge_regression_demo.py `):
``bash
python ridge_regression_demo.py
``

## Files

- `ridge_regression_demo.py `: A basic script with code for data generation, model implementation, training, evaluation, and visualization.
- `README.md `: This is the project description file.
