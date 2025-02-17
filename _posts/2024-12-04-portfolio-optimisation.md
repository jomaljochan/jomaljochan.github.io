---
title: Portfolio Optimisation
date: 2024-12-04 08:00:00 - 0000
categories: [London, Real Estate, Data Science, Analytics]
tags: [london, realestate, datascience, analytics]
#image: /path/to/image
alt: "Portfolio Optimisation"
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## **Table of Contents**
* TOC
{:toc}

## The Real-World Problem  

The global inflation rate on average has been rising ever since the 2020 pandemic with a forecast
of **6.52%** for 2023. The UK domestically faces a Consumer Price Index (CPI) of **10.1%**
in the year to January 2023. While inflation is deep in the double-figure territory, the base interest
rate published by the Bank of England is only at a bare 4.0%. Wages have grown at the fastest rate
in more than 20 years but are still failing to keep up with rising prices.

To confront this ongoing trend of rising of inflation, people need to find ways to invest their money rather than
locking it into savings accounts. We aim to build a ‘portfolio optimizer’ which can guide both non-savvy and savvy investors with their investment decisions. We hope to provide financial portfolio
ideas for people regardless of their risk-tolerance levels, time-frame preferences and investment
budgets. We have developed a prototype which works as a minimum viable portfolio optimizer
for our purposes. The prototype can be further improved and customized by adding more
preference inputs.

## Optimisation Formulation  

### Decision Variables  

We create an integer decision variable $$ x_{\text{a}} $$ for the amount of money invested in each asset. To
formulate an optimal portfolio, we select **6 investment opportunities** with different risk levels.
Bitcoin is chosen for high-risk investment. We also include safe investment such as bank savings,
gold and risk-free investments. The 6 decision variables are as follows:

$$
x_{\text{Bitcoin}}, \quad x_{\text{Bank}}, \quad x_{\text{FTSE}}, \quad x_{\text{Gold}}, \quad x_{\text{House}}, \quad x_{\text{Risk-Free}}
$$

### Constraints  

#### Minimum Return Constraint  

The portfolio return must meet or exceed the user-specified minimum return:

$$
\sum x_i (1 + r_i)^{\text{timeframe}} - \text{money invested} \geq \text{min _return}
$$

where $$ r_{\text{i}} $$ represents the return rate of asset **i**.  

#### Risk Constraint  

Portfolio risk is measured using **Modern Portfolio Theory** (Harry Markowitz Model) and incorporates **market risk** (0.000718) calculated from the **UK GDP index**:

$$
\sigma_p^2 = \sum x_i^2 \sigma_i^2 + 2 \sum x_i x_j \rho_{ij} \sigma_i \sigma_j + 0.000718
$$

which must satisfy:

$$
\sigma_p^2 \leq \text{max risk level}
$$

#### Budget Constraint  

The sum of investments should not exceed the available budget:

$$
x_{\text{Bitcoin}} + x_{\text{Bank}} + x_{\text{FTSE}} + x_{\text{Gold}} + x_{\text{House}} + x_{\text{Risk-Free}} \leq \text{money invested}
$$

#### Diversification Constraint  

To ensure diversification, we set a **maximum allocation limit** for any single security:

$$
x_i \leq \frac{1}{N} \times \text{money invested}
$$

where \( N \) is the number of assets in the portfolio.  

### Objective Function  

The objective is to **maximize total return** while considering the investment timeframe:

$$
\max \sum x_i (1 + r_i)^{\text{timeframe}}
$$

## Data Collection and Preprocessing  

Our model includes six asset classes: **risk-free assets, bank rates, FTSE, house prices, Bitcoin, and gold**. Monthly data from **January 2015 to December 2022** was sourced from Bloomberg, Statista, and Investing.com.  

- **Risk-Free Return**: Calculated as yearly return divided by **12**.  
- **Bank Rates**: Monthly returns derived from UK bank interest rates.  
- **FTSE Returns**: Includes monthly index returns and dividend payouts.  
- **House Prices, Bitcoin, and Gold**: Monthly average prices used to compute return variance and covariance.  

The processed data forms:  
- A **mean return dictionary**  
- A **variance dictionary**  
- A **covariance matrix**  

## Optimization Output and Decision Making  

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

Similarly, if **User B** prefers monthly investments of £800, the model suggests:
- **34.3%** in **House Prices**  `
- **34.3%** in **FTSE**  
- **28.2%** in **Bitcoin**  
- **3.18%** in **Gold**  

### Improvements After Initial Model  

Our **first model lacked a diversification constraint**, leading to the exclusion of certain asset classes. The revised version ensures diversified portfolios, accommodating various user preferences.  

We also incorporated a **line chart** displaying portfolio value projections under different **market conditions (poor, intermediate, and good)**.  

## Prototype Instructions and Further Improvements  

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

## Codebase

```python
# Import various packages
import pandas as pd
import numpy as np
import statistics
import folium # visualisation package for spatial data (plot of maps)
import seaborn as sns # general visualization package 
import matplotlib.pyplot as plt # general visualization package 
#next command allows you to display the figures in the notebook
%matplotlib inline    
```


```python
# Import the gurobi package
import gurobipy as gp
from gurobipy import GRB,quicksum
import datetime
```


```python
riskfree = pd.read_csv('data/riskfree.csv')
bitcoin = pd.read_csv('data/bitcoin.csv')
gold = pd.read_csv('data/Gold.csv')
ftse = pd.read_csv('data/FTSE.csv')
bank_rates = pd.read_excel('data/bank_rates.xlsx')
house_prices = pd.read_excel('data/house_prices.xlsx')
```


```python
riskfree = riskfree.dropna()
bitcoin = bitcoin.dropna()
gold = gold.dropna()
ftse = ftse.dropna()
bank_rates = bank_rates.dropna()
house_prices = house_prices.dropna()
```


```python
riskfree['Date'] = pd.to_datetime(riskfree['Date'], format='%b-%y')
riskfree = riskfree.sort_values(by='Date')
```


```python
bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])
bitcoin = bitcoin.sort_values(by="Date")
```


```python
gold['Date'] = pd.to_datetime(gold['Date'])
gold = gold.sort_values(by="Date")
```


```python
ftse['Date'] = pd.to_datetime(ftse['Date'])
ftse = ftse.sort_values(by="Date")
```


```python
bank_rates['Rate'] = bank_rates['Rate']/12
bank_rates
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-02-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-03-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-04-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-05-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2022-08-01</td>
      <td>0.145833</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2022-09-01</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2022-10-01</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2022-11-01</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2022-12-01</td>
      <td>0.291667</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 2 columns</p>
</div>




```python
house_prices['Date'] = pd.to_datetime(house_prices['Date'])
house_prices = house_prices.sort_values(by="Date")
```


```python
values = pd.DataFrame()
```


```python
# riskfree = pd.read_csv('riskfree.csv')
# bitcoin = pd.read_csv('bitcoin.csv')
# gold = pd.read_csv('Gold.csv')
# ftse = pd.read_csv('FTSE.csv')
# bank_rates = pd.read_excel('bank_rates.xlsx')
# house_prices = pd.read_excel('house_prices.xlsx')
bank_rates
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-02-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-03-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-04-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-05-01</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2022-08-01</td>
      <td>0.145833</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2022-09-01</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2022-10-01</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2022-11-01</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2022-12-01</td>
      <td>0.291667</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 2 columns</p>
</div>




```python
values.index = riskfree['Date'].dt.strftime('%m/%Y')
values['riskfree'] = riskfree['Price'].values
values['bitcoin'] = bitcoin['Open'].values
values["gold"] = gold['Price'].values
values["ftse"] = ftse['Price'].values
values['ftse'] = values['ftse'].str.replace(',','')
values["ftse"] = values["ftse"].astype(float)
values["house_prices"] = house_prices['PX_MID'].values
values["bank_rates"] = bank_rates['Rate'].values
```


```python
values
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>riskfree</th>
      <th>bitcoin</th>
      <th>gold</th>
      <th>ftse</th>
      <th>house_prices</th>
      <th>bank_rates</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01/2015</th>
      <td>1.341</td>
      <td>320.434998</td>
      <td>441.71</td>
      <td>6749.40</td>
      <td>190665</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>02/2015</th>
      <td>1.789</td>
      <td>216.867004</td>
      <td>418.89</td>
      <td>6946.66</td>
      <td>190827</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>03/2015</th>
      <td>1.579</td>
      <td>254.283005</td>
      <td>408.56</td>
      <td>6773.04</td>
      <td>191537</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>04/2015</th>
      <td>1.835</td>
      <td>244.223007</td>
      <td>408.29</td>
      <td>6960.63</td>
      <td>193225</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>05/2015</th>
      <td>1.804</td>
      <td>235.938995</td>
      <td>410.84</td>
      <td>6984.43</td>
      <td>195313</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>08/2022</th>
      <td>2.799</td>
      <td>23336.718750</td>
      <td>596.06</td>
      <td>7284.15</td>
      <td>292206</td>
      <td>0.145833</td>
    </tr>
    <tr>
      <th>09/2022</th>
      <td>4.096</td>
      <td>20050.498050</td>
      <td>577.35</td>
      <td>6893.81</td>
      <td>294274</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>10/2022</th>
      <td>3.523</td>
      <td>19431.105470</td>
      <td>566.54</td>
      <td>7094.53</td>
      <td>294996</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>11/2022</th>
      <td>3.163</td>
      <td>20494.898440</td>
      <td>607.70</td>
      <td>7573.05</td>
      <td>295608</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>12/2022</th>
      <td>3.669</td>
      <td>17168.001950</td>
      <td>630.59</td>
      <td>7451.74</td>
      <td>294329</td>
      <td>0.291667</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 6 columns</p>
</div>




```python
returns = pd.DataFrame()
returns = values.pct_change(1)
returns['bank_rates'] = values['bank_rates']
```


```python
returns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>riskfree</th>
      <th>bitcoin</th>
      <th>gold</th>
      <th>ftse</th>
      <th>house_prices</th>
      <th>bank_rates</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01/2015</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>02/2015</th>
      <td>0.334079</td>
      <td>-0.323211</td>
      <td>-0.051663</td>
      <td>0.029226</td>
      <td>0.000850</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>03/2015</th>
      <td>-0.117384</td>
      <td>0.172530</td>
      <td>-0.024660</td>
      <td>-0.024993</td>
      <td>0.003721</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>04/2015</th>
      <td>0.162128</td>
      <td>-0.039562</td>
      <td>-0.000661</td>
      <td>0.027697</td>
      <td>0.008813</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>05/2015</th>
      <td>-0.016894</td>
      <td>-0.033920</td>
      <td>0.006246</td>
      <td>0.003419</td>
      <td>0.010806</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>08/2022</th>
      <td>0.510523</td>
      <td>0.177405</td>
      <td>-0.031206</td>
      <td>-0.018762</td>
      <td>0.011072</td>
      <td>0.145833</td>
    </tr>
    <tr>
      <th>09/2022</th>
      <td>0.463380</td>
      <td>-0.140818</td>
      <td>-0.031389</td>
      <td>-0.053588</td>
      <td>0.007077</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>10/2022</th>
      <td>-0.139893</td>
      <td>-0.030892</td>
      <td>-0.018723</td>
      <td>0.029116</td>
      <td>0.002453</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>11/2022</th>
      <td>-0.102186</td>
      <td>0.054747</td>
      <td>0.072652</td>
      <td>0.067449</td>
      <td>0.002075</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>12/2022</th>
      <td>0.159975</td>
      <td>-0.162328</td>
      <td>0.037667</td>
      <td>-0.016019</td>
      <td>-0.004327</td>
      <td>0.291667</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 6 columns</p>
</div>




```python
mean_returns = returns.mean()
mean_returns = mean_returns.to_dict()
```


```python
mean_returns['riskfree'] = 0.00322 # current yearly rate is 3.864%, thus monthly rate is 3.864/12
mean_returns
```




    {'riskfree': 0.00322,
     'bitcoin': 0.06687589620239774,
     'gold': 0.004493369767621386,
     'ftse': 0.0017219801309813312,
     'house_prices': 0.004640390645573267,
     'bank_rates': 0.045876736111111115}




```python
variance = returns.var()
variance = variance.to_dict()
```


```python
variance['riskfree'] = 0 # assume
variance
```




    {'riskfree': 0,
     'bitcoin': 0.05276732797060849,
     'gold': 0.001509312501510058,
     'ftse': 0.0013549727170874425,
     'house_prices': 0.00012109719949019301,
     'bank_rates': 0.002178543418006822}




```python
covariance = values.cov()
```


```python
covariance['riskfree'] = 0
covariance.iloc[0] = 0
covariance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>riskfree</th>
      <th>bitcoin</th>
      <th>gold</th>
      <th>ftse</th>
      <th>house_prices</th>
      <th>bank_rates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>riskfree</th>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bitcoin</th>
      <td>0</td>
      <td>2.652161e+08</td>
      <td>1.174646e+06</td>
      <td>1.890792e+06</td>
      <td>3.215854e+08</td>
      <td>-19.239194</td>
    </tr>
    <tr>
      <th>gold</th>
      <td>0</td>
      <td>1.174646e+06</td>
      <td>9.018851e+03</td>
      <td>-3.173816e+03</td>
      <td>2.008619e+06</td>
      <td>0.315971</td>
    </tr>
    <tr>
      <th>ftse</th>
      <td>0</td>
      <td>1.890792e+06</td>
      <td>-3.173816e+03</td>
      <td>2.873414e+05</td>
      <td>4.278158e+06</td>
      <td>7.791421</td>
    </tr>
    <tr>
      <th>house_prices</th>
      <td>0</td>
      <td>3.215854e+08</td>
      <td>2.008619e+06</td>
      <td>4.278158e+06</td>
      <td>6.936991e+08</td>
      <td>567.095727</td>
    </tr>
    <tr>
      <th>bank_rates</th>
      <td>0</td>
      <td>-1.923919e+01</td>
      <td>3.159706e-01</td>
      <td>7.791421e+00</td>
      <td>5.670957e+02</td>
      <td>0.002179</td>
    </tr>
  </tbody>
</table>
</div>



### User-defined input


```python
#vivan's code - linked to extract the values that were user's input from the dashboard

#our temporary input
# time_frame = 2

# amount_invested = 1000 

# min_return = 5000

# max_risk = 0.1

time_frame = 12
min_return = 5000
max_risk = 10
amount_invested = 100000
```

### Model creation


```python
# Create a new model:
m = gp.Model("portfolio")
```

    Set parameter Username
    Academic license - for non-commercial use only - expires 2024-02-09


### Data

*Define the dictionaries containing the data:*


```python
# Fixed inputs
assets = ['riskfree', 'bitcoin', 'gold', 'ftse', 'house_prices', 'bank_rates'] 

# will need to get real data from Ishaan to substitute below:
returns = mean_returns

risks = variance
risks
```




    {'riskfree': 0,
     'bitcoin': 0.05276732797060849,
     'gold': 0.001509312501510058,
     'ftse': 0.0013549727170874425,
     'house_prices': 0.00012109719949019301,
     'bank_rates': 0.002178543418006822}



### Optimal allocation strategy


1. **Decision variables:** We create a decision variable $x_a$ for the amount invested for each asset $a \in {\rm assets}$.
This means that we have the following decision variables: $x_{crypto}, x_{real estate}, x_{FTSE},x_{tech stocks}, x_{savings account}, x_{bond 3m}, x_{bond 1 yr}, x_{bond 5 yr},x_{bond 10 yr}$. The quantity of the optimal budget allocation over assets is unknown.


2. **Constraints:** We need to ensure that the total budget spent across all the keywords does not exceed the fixed amount to be invested that was entered by the user. Mathematically, this requirement is expressed using a linear constraint: 
<br><br>
$$  x_{crypto} + x_{real estate} + x_{FTSE} + x_{tech stocks} + x_{savings account} + x_{bond 3m}+x_{bond 1 yr}+x_{bond 5 yr}+x_{bond 10 yr} \leq \quad money\quad invested$$
<br>
In this constraint, the total money invested spent should not exceed the restriction of the user input.

    
3. **Objective function:** Now, we need to select the objective function. *What should we optimize for?* The goal is find a combination of money allocation for each asset that maximizes the final return. It is shown with the following expression:

<br><br>
$$  \max_{x_a}  \quad\quad\quad  x_{crypto}\cdot(1+return rate_{crypto})^{time frame} + x_{real estate}\cdot(1+return rate_{real estate})^{time frame} + x_{FTSE}\cdot(1+return rate_{FTSE})^{time frame} + x_{tech stocks}\cdot(1+return rate_{tech stocks})^{time frame} + x_{savings account}\cdot(1+return rate_{savings account})^{time frame} + x_{bond 3m}\cdot(1+return rate_{bond 3m})^{time frame} + x_{bond 1 yr}\cdot(1+return rate_{bond 1 yr})^{time frame} + x_{bond 5 yr}\cdot(1+return rate_{bond 5 yr})^{time frame} + x_{bond 10 yr}\cdot(1+return rate_{bond 10 yr})^{time frame}$$
<br>

In this function, we maximize the return for a given time horizon specified by the user (ie. *time_frame* refers to the time period for which the user is ready to have their money locked in the investment portfolio). 

Shortly, we can express the function as following:

$$ \max_{x_a} \quad\quad\quad \sum_{a=1}^{n} X_a \cdot (1+return rate_a)^{time frame}$$


### Decision variables


```python
investment_amount = m.addVars(assets, vtype=GRB.INTEGER, lb = 0, name = "investment_amount")
```

### Constraints 

Add constraints for:

1) minimum output return that is accepted (ie. that the output should be greater than or equal to the return desired by the user)

2) maximum level of risk accepted (ie. that the output portfolio should have risk that does not exceed the maximum risk specified by the user)


```python
#proxy values
# time_frame = 10
# min_return = 0.03
# max_risk = 0.1
# amount_invested = 10000

#min return accepted       
m.addConstr((quicksum(investment_amount[a]*((1+returns[a])**(12*time_frame)) for a in assets)-amount_invested >= min_return),
             name = "minimum return accepted")

#max risk accepted
# m.addConstr((quicksum(investment_amount[a1]*investment_amount[a2]*risks[a1]*risks[a2]*covariance.loc[a1,a2]/((amount_invested)**2) 
#                        for a1 in assets for a2 in assets)) <= max_risk, name="maximum risk accepted")

#max risk accepted corrected
m.addConstr((quicksum(investment_amount[a1]*investment_amount[a2]*covariance.loc[a1,a2]/((amount_invested)**2) 
                       for a1 in assets for a2 in assets)) <= (max_risk**2), name="maximum risk accepted")

#sum of investments
m.addConstr((quicksum(investment_amount[a1] for a1 in assets)) == amount_invested, name="sum of investments")
```




    <gurobi.Constr *Awaiting Model Update*>



### Objective

Formulate the objective function 


```python
# Objective function:
m.setObjective(quicksum(investment_amount[a]*((1+returns[a])**(12*time_frame)) for a in assets), 
              GRB.MAXIMIZE)

#add the objective function to minimize risk 
#send Vivian the output from running the model
```

### Solve

After having formulated and implemented the integer program, we can now optimize the portfolio allocation and printout the optimal return:


```python
# Run the optimization
def printSolution():
    if m.status == GRB.OPTIMAL:
        print('\nPortfolio Return: %g' % m.objVal)
        print('\nInvestment Amount:')
        investment_amountx = m.getAttr('x', investment_amount) 
        for a in assets:            
                print('%s %g' % (a, investment_amountx[a]))
    else:
        print('No solution:', m.status)
        
m.optimize()
printSolution()
```

    Gurobi Optimizer version 10.0.0 build v10.0.0rc2 (mac64[rosetta2])
    
    CPU model: Apple M1 Pro
    Thread count: 8 physical cores, 8 logical processors, using up to 8 threads
    
    Optimize a model with 2 rows, 6 columns and 12 nonzeros
    Model fingerprint: 0xdad02658
    Model has 1 quadratic constraint
    Variable types: 0 continuous, 6 integer (0 binary)
    Coefficient statistics:
      Matrix range     [1e+00, 1e+04]
      QMatrix range    [2e-13, 7e-02]
      Objective range  [1e+00, 1e+04]
      Bounds range     [0e+00, 0e+00]
      RHS range        [1e+05, 1e+05]
      QRHS range       [1e+02, 1e+02]
    Presolve removed 1 rows and 0 columns
    Presolve time: 0.00s
    Presolved: 1 rows, 6 columns, 6 nonzeros
    Presolved model has 1 quadratic constraint(s)
    Variable types: 0 continuous, 6 integer (0 binary)
    
    Root relaxation: objective 6.453976e+07, 1 iterations, 0.00 seconds (0.00 work units)
    
        Nodes    |    Current Node    |     Objective Bounds      |     Work
     Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time
    
         0     0 6.4540e+07    0    -          - 6.4540e+07      -     -    0s
    *    0     0               0    6.449760e+07 6.4498e+07 -0.00%     -    0s
    
    Explored 1 nodes (1 simplex iterations) in 0.01 seconds (0.00 work units)
    Thread count was 8 (of 8 available processors)
    
    Solution count 1: 6.44976e+07 
    
    Optimal solution found (tolerance 1.00e-04)
    Best objective 6.449760003949e+07, best bound 6.449760003949e+07, gap 0.0000%
    
    Portfolio Return: 6.44976e+07
    
    Investment Amount:
    riskfree -0
    bitcoin 61
    gold -0
    ftse -0
    house_prices -0
    bank_rates 99939



```python
time_frame
```




    12




```python
risks
```




    {'riskfree': 0,
     'bitcoin': 0.05276732797060849,
     'gold': 0.001509312501510058,
     'ftse': 0.0013549727170874425,
     'house_prices': 0.00012109719949019301,
     'bank_rates': 0.002178543418006822}




```python
returns
```




    {'riskfree': 0.00322,
     'bitcoin': 0.06687589620239774,
     'gold': 0.004493369767621386,
     'ftse': 0.0017219801309813312,
     'house_prices': 0.004640390645573267,
     'bank_rates': 0.045876736111111115}




```python
returns.median()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[34], line 1
    ----> 1 returns.median()


    AttributeError: 'dict' object has no attribute 'median'



```python
returns.mean()
```




    riskfree        0.047547
    bitcoin         0.066876
    gold            0.004493
    ftse            0.001722
    house_prices    0.004640
    bank_rates      0.045877
    dtype: float64




```python
import numpy as np
import pandas as pd
df_line = pd.DataFrame({
        "Years": range(0,10), 
        "Poor market conditions": np.random.rand(10), 
        "Intermediate market conditions":  np.random.rand(10), 
        "Good market conditions": np.random.rand(10), 
})

df_line = df_line.set_index('Years').stack().reset_index()
df_line.rename({'level_1':'Market Conditions', 0:"Value"}, inplace=True, axis=1)
print(df_line.head())
```

       Years               Market Conditions     Value
    0      0          Poor market conditions  0.976242
    1      0  Intermediate market conditions  0.560914
    2      0          Good market conditions  0.600379
    3      1          Poor market conditions  0.924869
    4      1  Intermediate market conditions  0.082201

## Conclusion

