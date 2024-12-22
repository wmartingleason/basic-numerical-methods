# basic-numerical-methods

This repository contains implementations of five numerical methods for solving ordinary differential equations. The methods implemented are:
    Euler's Method,
    Improved Euler's Method,
    Fourth-Order Runge-Kutta Method,
    Fourth-Order Adams-Bashforth Method,
    Fourth-Order Predictor-Corrector Method (using the Adams-Bashforth method as the predictor and the Adams-Moulton method as the corrector)

The code also includes functions for calculating Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for each method to help analyze the performance of these methods with different parameters and initial value problems.

Purpose
I am building this project as a study tool while working through Chapter 8 of Elementary Differential Equations and Boundary Value Problems by Boyce, DiPrima, and Meade. If you are learning about numerical methods for solving ODEs, this code will allow you to experiment with and compare the error of different methods for a variety of problems.

To use this code, clone the repository to your local machine:

git clone https://github.com/wmartingleason/basic-numerical-methods.git
cd basic-numerical-methods

The repository requires Python 3.x and NumPy. It is licensed under the MIT License.
