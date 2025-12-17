# Analysis Results Interpretation Guide

This guide explains how to interpret the results from the macroparameter analysis script and what they mean for your inverse mapping problem (measurements -> macroparameters).

---

## 1. CORRELATION ANALYSIS

### What is Correlation?
Correlation measures the **strength and direction** of the linear relationship between two variables.

### Correlation Coefficient (Pearson's r)
- **Range**: -1 to +1
- **+1**: Perfect positive correlation (as macro increases, measurement increases)
- **-1**: Perfect negative correlation (as macro increases, measurement decreases)
- **0**: No linear correlation

### Interpretation Thresholds:
- **|r| > 0.9**: Very strong correlation
- **|r| 0.7-0.9**: Strong correlation
- **|r| 0.5-0.7**: Moderate correlation
- **|r| 0.3-0.5**: Weak correlation
- **|r| < 0.3**: Very weak/no correlation

### What This Tells You:
The correlation heatmap shows which macroparameters directly affect which measurements:

**Example:**
```
height <-> height_cm: r = +0.95
```
This means the `height` macroparameter has a very strong positive effect on `height_cm`.

**Why This Matters:**
- Strong correlations (|r| > 0.7) indicate which macroparameters are **most important** for controlling specific measurements
- For inverse mapping, focus on macroparameters with the strongest correlations to your target measurements

---

## 2. REGRESSION MODELS

Regression models predict measurements from macroparameters. We test three types to understand relationship complexity.

### Model Types:

#### A. Linear Regression
- **Assumes**: Simple linear relationships (y = ax + b)
- **Best for**: Straightforward, proportional relationships
- **Equation example**: `height_cm = 50 * height + 30 * age + 100`

#### B. Polynomial Regression (degree=2)
- **Assumes**: Quadratic relationships and interactions
- **Best for**: Curved relationships and parameter interactions
- **Equation example**: `height_cm = 50*height + 30*age + 5*height^2 + 10*height*age + 100`
- **Captures**: Non-linear effects and how parameters work together

#### C. Random Forest
- **Assumes**: Complex, non-linear relationships
- **Best for**: Highly complex patterns
- **How it works**: Builds decision trees to capture any relationship shape

---

## 3. PERFORMANCE METRICS

### R² Score (R-squared)
**What it measures**: How well the model fits the data (goodness of fit)

- **Range**: 0 to 1 (can be negative for very poor models)
- **1.0**: Perfect predictions (100% variance explained)
- **0.9+**: Excellent fit (90%+ variance explained)
- **0.7-0.9**: Good fit
- **0.5-0.7**: Moderate fit
- **< 0.5**: Poor fit

**Interpretation:**
```
shoulder_width_cm: R² = 0.95
```
The model explains 95% of the variation in shoulder width. Very reliable!

### MAE (Mean Absolute Error)
**What it measures**: Average prediction error in centimeters

**Example:**
```
shoulder_width_cm: MAE = 0.35 cm
```
On average, predictions are off by ±0.35 cm. Pretty accurate!

**Interpretation:**
- **MAE < 0.5 cm**: Excellent accuracy
- **MAE 0.5-1.0 cm**: Good accuracy
- **MAE 1.0-2.0 cm**: Moderate accuracy
- **MAE > 2.0 cm**: Poor accuracy

### RMSE (Root Mean Squared Error)
**What it measures**: Similar to MAE, but penalizes larger errors more heavily

**Why use both MAE and RMSE?**
- If RMSE >> MAE: Model has some large errors
- If RMSE ≈ MAE: Errors are consistent

---

## 4. COMPARING MODELS

### What to Look For:

#### Case 1: Linear Model Works Well
```
Linear R²: 0.95
Polynomial R²: 0.96
Random Forest R²: 0.96
```
**Interpretation:** Relationship is **primarily linear**
**For inverse mapping:** Simple linear equations will work!

#### Case 2: Polynomial Model Improves Significantly
```
Linear R²: 0.75
Polynomial R²: 0.92
Random Forest R²: 0.93
```
**Interpretation:** Relationship has **polynomial components** and **interactions**
**For inverse mapping:** Need to consider parameter interactions (e.g., height × proportions)

#### Case 3: Random Forest Much Better
```
Linear R²: 0.65
Polynomial R²: 0.72
Random Forest R²: 0.95
```
**Interpretation:** Relationship is **highly non-linear and complex**
**For inverse mapping:** May need optimization algorithms or machine learning for inverse

---

## 5. FEATURE IMPORTANCE (Random Forest)

### What it Shows:
How much each macroparameter contributes to predicting each measurement.

**Example:**
```
height_cm:
  height:      0.850
  age:         0.095
  proportions: 0.030
  weight:      0.015
  muscle:      0.010
```

**Interpretation:**
- `height` macroparameter accounts for 85% of the prediction power for height_cm
- Other parameters have minor effects

### Why This Matters:
- **Prioritize important parameters**: When doing inverse mapping, focus on parameters with high importance
- **Reduce complexity**: Parameters with very low importance (<0.05) might be set to default values
- **Understand control**: Tells you which "knobs" to adjust for specific measurements

---

## 6. INTERACTION EFFECTS

### What They Show:
How pairs of macroparameters work **together** to affect measurements.

**Example visualization:**
```
height × proportions -> height_cm
```
The scatter plot shows if the combination of height and proportions creates patterns.

### How to Interpret:

#### Strong Interaction:
If you see clear diagonal bands or curved patterns in color, the two parameters interact significantly.

**Example:** "When height is high, proportions has a stronger effect on shoulder width"

#### Weak Interaction:
If colors are evenly distributed (horizontal/vertical bands only), parameters work independently.

---

## 7. PRACTICAL APPLICATION TO YOUR PROBLEM

### Your Goal: Measurements -> Macroparameters

#### Step 1: Identify Key Relationships
Look at correlation matrix:
```
height <-> height_cm: r = 0.95
proportions <-> shoulder_width_cm: r = 0.82
```

#### Step 2: Check Model Complexity
If linear R² is high (>0.9): You can use simple algebraic inversion!

**Example:**
```
height_cm = 50 * height + 100
```
**Inverse:**
```
height = (height_cm - 100) / 50
```

#### Step 3: Feature Importance for Multi-Parameter Measurements
For complex measurements (affected by multiple macros):

**Example: shoulder_width_cm**
```
Feature importance:
  proportions: 0.45
  weight: 0.35
  height: 0.15
  muscle: 0.05
```

**Strategy:** Use optimization focusing on proportions and weight first.

#### Step 4: Handle Interactions
If polynomial model is much better than linear, you need to solve for interactions:

**Example:**
```
shoulder_width = a*proportions + b*weight + c*proportions*weight + d
```

Use optimization algorithms like:
- **scipy.optimize.minimize**: For smooth optimization
- **Grid search**: For discrete parameter spaces
- **Gradient descent**: For differentiable models

---

## 8. DECISION TREE FOR YOUR INVERSE MAPPING

```
START: You have panellist measurements

├─ Is R² > 0.9 for linear models?
│  ├─ YES -> Use algebraic inversion (simple formulas)
│  └─ NO -> Continue
│
├─ Is polynomial R² >> linear R²?
│  ├─ YES -> Account for parameter interactions
│  └─ NO -> Continue
│
└─ Random Forest R² >> others?
   ├─ YES -> Use optimization-based inverse mapping
   └─ NO -> Data may have issues, investigate further
```

---

## 9. EXAMPLE INTERPRETATION

Let's say the analysis shows:

### Results:
```
height_cm:
  Linear R²: 0.98
  Polynomial R²: 0.98
  Random Forest R²: 0.98
  Top parameter: height (importance: 0.95)

shoulder_width_cm:
  Linear R²: 0.75
  Polynomial R²: 0.92
  Random Forest R²: 0.93
  Top parameters: proportions (0.50), weight (0.30), height (0.15)
```

### What This Means:

**For height_cm:**
- Simple linear relationship with `height` macroparameter
- Can use direct formula: `height_macro = f(height_cm)`
- Very predictable and accurate

**For shoulder_width_cm:**
- More complex relationship
- Multiple macroparameters involved
- Polynomial/interaction terms matter
- Need optimization approach that considers proportions, weight, and height together
- May need to accept some error (MAE will tell you how much)

---

## 10. NEXT STEPS AFTER ANALYSIS

Based on results, you'll implement:

### If Linear Works (R² > 0.9):
```python
# Direct algebraic inverse
height_macro = (height_cm - intercept) / coefficient
```

### If Polynomial Works Better:
```python
# Optimization with polynomial constraints
from scipy.optimize import minimize

def objective(macros):
    predicted = polynomial_model.predict(macros)
    return sum((predicted - target_measurements)**2)

result = minimize(objective, initial_guess, bounds=macro_bounds)
```

### If Highly Non-linear:
```python
# ML-based inverse mapping
# Train a model: measurements -> macroparameters
inverse_model = RandomForestRegressor()
inverse_model.fit(measurements, macroparameters)

predicted_macros = inverse_model.predict(panellist_measurements)
```

---

## SUMMARY

**The analysis tells you:**
1. **Which macroparameters control which measurements** (correlation)
2. **How complex the relationships are** (model comparison)
3. **How accurate predictions can be** (R², MAE, RMSE)
4. **Which parameters matter most** (feature importance)
5. **Whether parameters interact** (interaction effects)

**This guides your inverse mapping strategy:**
- Simple relationships -> Direct formulas
- Moderate complexity -> Polynomial optimization
- High complexity -> Machine learning or advanced optimization

**The goal:** Find the macroparameter values that produce a mesh matching your panellist measurements as closely as possible!
