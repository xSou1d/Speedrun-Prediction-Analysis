# \# Speedrun Performance Prediction

# 

# A data analysis and predictive modeling study using real-world speedrun data from \[Speedrun.com](https://www.speedrun.com/). The project investigates how player performance evolves across repeated attempts and evaluates which functional form best captures the learning curve structure in speedrunning data.

# 

# \---

# 

# \## Research Question

# 

# Do speedrun completion times follow a predictable decay pattern as a runner accumulates attempts? And if so, which model — linear, polynomial, or exponential decay — best describes that pattern?

# 

# \---

# 

# \## Dataset

# 

# \- \*\*Source:\*\* Speedrun.com (manual export)

# \- \*\*Observations:\*\* 295 run records

# \- \*\*Subjects:\*\* 65 individual runners

# \- \*\*Task categories:\*\* 10 distinct game/category combinations

# \- \*\*Features:\*\* attempt number, completion time, runner ID, category

# 

# \---

# 

# \## Methodology

# 

# \### Model Family Evaluated

# 

# Three functional forms were constructed and compared:

# 

# | Model | Form | Hypothesis |

# |-------|------|------------|

# | Linear | `t = a - b\*x` | Constant improvement per attempt |

# | Polynomial | `t = a + bx + cx²` | Diminishing but continuous improvement |

# | Exponential decay | `t = a \* e^(-bx) + c` | Rapid early gains, asymptotic floor |

# 

# \### Model Selection Process

# 

# 1\. Fit each model to the training split using `scipy.optimize.curve\_fit`

# 2\. Inspect residual plots for systematic patterns (violation of homoscedasticity)

# 3\. Compare out-of-sample fit using RMSE on the held-out test split

# 4\. Test assumptions: normality of residuals, independence, absence of trend in residuals

# 

# \### Lookahead / Data Leakage Prevention

# 

# Model selection was performed exclusively on the training split. The test split was held out until final evaluation — no parameters were tuned using test data.

# 

# \---

# 

# \## Results

# 

# \*\*Exponential decay was selected as the superior model\*\* across all evaluation criteria:

# 

# \- Lowest out-of-sample RMSE

# \- Residuals showed no systematic trend (linear and polynomial both exhibited curvature in residual plots)

# \- Consistent with theoretical reasoning: early attempts yield large time improvements; improvements compress as performance approaches a physiological or execution ceiling

# 

# \*\*Key finding:\*\* The asymptotic parameter `c` in the exponential model estimates the theoretical performance floor — the minimum achievable time given the runner's current routing and execution strategy. This has a natural interpretation: no runner improves indefinitely, and the model captures that boundary explicitly.

# 

# \---

# 

# \## Project Structure

# 

# ```

# Speedrun-Prediction-Analysis/

# ├── SpeedrunAnalysis.py   # Full analysis pipeline

# ├── game\_data.csv         # Raw run records

# ├── analysis.csv          # Processed output with model predictions

# └── README.md

# ```

# 

# \---

# 

# \## Collaboration Note

# 

# This project was completed as a two-person academic final project. My contributions were the core modeling pipeline (model family evaluation, nonlinear curve fitting, model selection methodology) and the majority of visualizations. My partner contributed the correlation analysis, improvement rate statistics, the Power Law Fit visualization, and led the written analysis. The README and modeling code reflect my individual work within the collaboration.

# 

# \---

# 

# \## How to Run

# 

# ```bash

# pip install pandas numpy matplotlib scipy

# python SpeedrunAnalysis.py

# ```

# 

# Outputs residual plots, fitted curves, and a summary table of model comparison metrics.

# 

# \---

# 

# \## Skills Demonstrated

# 

# \- Empirical model selection using residual analysis and out-of-sample evaluation

# \- Nonlinear curve fitting with `scipy.optimize`

# \- Assumption testing for regression models

# \- Interpretation of asymptotic model parameters in a real-world context

# \- Clean, reproducible data pipeline with documented methodology

