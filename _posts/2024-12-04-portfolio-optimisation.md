---
title: Portfolio Optimisation
date: 2024-12-04 08:00:00 - 0000
categories: [London, Real Estate, Data Science, Analytics]
tags: [london, realestate, datascience, analytics]
#image: /path/to/image
alt: "Portfolio Optimisation"
---

# Portfolio Optimizer Project

## 1. The Real-World Problem  

The global inflation rate has been rising since the 2020 pandemic, with a forecast of **6.52%** for 2023. The UK faces a **Consumer Price Index (CPI) of 10.1%** in January 2023, while the Bank of England's base interest rate is only **4.0%**. Despite wages growing at the fastest rate in over 20 years, they still fail to keep up with inflation.  

To tackle inflation, people must explore investment options rather than relying on savings accounts. We aim to build a **portfolio optimizer** to assist both novice and experienced investors in making informed investment decisions. The prototype we developed serves as a **minimum viable product** and can be enhanced by incorporating additional preference inputs.  

## 2. Optimization Formulation  

### Decision Variables  

We define an integer decision variable \( x_a \) for the amount of money allocated to each asset. Our model includes **six investment options** with different risk levels:  

\[
x_{\text{Bitcoin}}, \quad x_{\text{Bank}}, \quad x_{\text{FTSE}}, \quad x_{\text{Gold}}, \quad x_{\text{House}}, \quad x_{\text{Risk-Free}}
\]

### Constraints  

#### Minimum Return Constraint  

The portfolio return must meet or exceed the user-specified minimum return:

\[
\sum x_i (1 + r_i)^{\text{timeframe}} - \text{money invested} \geq \text{min\_return}
\]

where \( r_i \) represents the return rate of asset \( i \).  

#### Risk Constraint  

Portfolio risk is measured using **Modern Portfolio Theory** (Harry Markowitz Model) and incorporates **market risk** (\( 0.000718 \)) calculated from the **UK GDP index**:

\[
\sigma_p^2 = \sum x_i^2 \sigma_i^2 + 2 \sum x_i x_j \rho_{ij} \sigma_i \sigma_j + 0.000718
\]

which must satisfy:

\[
\sigma_p^2 \leq \text{max risk level}
\]

#### Budget Constraint  

The sum of investments should not exceed the available budget:

\[
x_{\text{Bitcoin}} + x_{\text{Bank}} + x_{\text{FTSE}} + x_{\text{Gold}} + x_{\text{House}} + x_{\text{Risk-Free}} \leq \text{money invested}
\]

#### Diversification Constraint  

To ensure diversification, we set a **maximum allocation limit** for any single security:

\[
x_i \leq \frac{1}{N} \times \text{money invested}
\]

where \( N \) is the number of assets in the portfolio.  

### Objective Function  

The objective is to **maximize total return** while considering the investment timeframe:

\[
\max \sum x_i (1 + r_i)^{\text{timeframe}}
\]

## 3. Data Collection and Preprocessing  

Our model includes six asset classes: **risk-free assets, bank rates, FTSE, house prices, Bitcoin, and gold**. Monthly data from **January 2015 to December 2022** was sourced from Bloomberg, Statista, and Investing.com.  

- **Risk-Free Return**: Calculated as yearly return divided by **12**.  
- **Bank Rates**: Monthly returns derived from UK bank interest rates.  
- **FTSE Returns**: Includes monthly index returns and dividend payouts.  
- **House Prices, Bitcoin, and Gold**: Monthly average prices used to compute return variance and covariance.  

The processed data forms:  
- A **mean return dictionary**  
- A **variance dictionary**  
- A **covariance matrix**  

## 4. Optimization Output and Decision Making  

The model provides two key outputs:  

1. **Maximum achievable portfolio return**  
2. **Optimal investment allocation**  

For example, if **User A** has the following preferences:  

- **Investment horizon:** 10 years  
- **Initial investment:** £10,000  
- **Target return:** £13,000 (30% expected return)  
- **Risk level:** Medium  
- **Minimum diversification:** 3 assets  

Our model suggests an **optimal allocation**:  
- **33.9%** in **Bank Rates**  
- **29.6%** in **Risk-Free**  
- **28.2%** in **Bitcoin**  
- **8.06%** in **House Prices**  

Similarly, if **User B** prefers monthly investments of £800, the model suggests:**  
- **34.3%** in **House Prices**  
- **34.3%** in **FTSE**  
- **28.2%** in **Bitcoin**  
- **3.18%** in **Gold**  

### Improvements After Initial Model  

Our **first model lacked a diversification constraint**, leading to the exclusion of certain asset classes. The revised version ensures diversified portfolios, accommodating various user preferences.  

We also incorporated a **line chart** displaying portfolio value projections under different **market conditions (poor, intermediate, and good)**.  

## 5. Prototype Instructions and Further Improvements  

To use our dashboard, users provide **five inputs**:  

1. **Risk Preference**: "Moderate" or "Conservative"  
2. **Monthly Investment Amount** (max £10,000)  
3. **Expected Return** (max £10M)  
4. **Investment Time Horizon** (0–80 years)  
5. **Scenario Analysis for Market Conditions**  

### Future Enhancements  

- **Multi-currency support**  
- **"Advanced Mode" for professional investors**, allowing fine-tuned risk selection  
- **Additional asset classes** for better portfolio customization  

## Appendix  

### Graph 1 - Inflation Trends (IMF Data)  

*(Insert Graph Here)*  

### Graph 2 - UK Wage Growth vs Inflation  

*(Insert Graph Here)*  

### Graph 3 - Portfolio Allocation for One-Time Investment  

*(Insert Graph Here)*  

### Graph 4 - Portfolio Allocation for Monthly Investments  

*(Insert Graph Here)*  

---

This Markdown file correctly formats your equations using **LaTeX** with MathJax for Jekyll, ensuring they render properly on your site. You can embed the graphs manually where indicated.

Let me know if you need further refinements! 🚀
