---
title: Brand Value Analysis
date: 2025-01-03 08:00:00 - 0000
categories: [Data Science, Analytics, AI, Marketing, Branding]
tags: [datascience, analytics, marketing, branding]
#image: /path/to/image
alt: "Brand Value Analysis"
---

**Objective:** Evaluate the value of three personal care brands for Curve and recommend service fee adjustments.


## Executive Summary
### Most Valuable Brand:
- **Brand 3** is the most valuable brand as it has the highest Lifetime Value over the next year
- Although the Operating Profit from Brand 2 is higher than that of Brand 3, the much higher growth rate along with the high service fee makes Brand 3 almost 3x the Lifetime Value of Brand 2

### Potential impactful factors:
**1. Data Completeness & Confidence**
   - **Variance & Confidence Metrics**: Lack of variance data limits confidence, especially in growth rate estimates.
   - **Customer Distribution & Loyalty**: Limited data on customer overlap and loyalty can lead to unpredictable sales cannibalisation.
   - **Fixed Costs Exclusion**: Ignoring fixed costs may undervalue real-world profitability.

**2. Market Dynamics & Seasonality**
   - **Seasonality Ignored**: Ignoring seasonality could misrepresent actual revenue patterns, impacting predictions.
   - **Market Saturation & Customer Mix**: Uncertain market conditions and customer base mix may affect growth stability and ROI.

**3. Operational Assumptions**
   - **Assumed Weekly Operating Cycle**: Simplified weekly cycle may not align with real 30–60 day retail cash flow cycles, impacting LTV accuracy.

### Fee Recommendation & negotiations:
- Primary action is to increase Brand 3 Service Fee to 30% : As growth is high, churn will be nullified.
- Brand 1 & 2 service fee to be increased to 21% and 26% respectively to test impact
- To review all brands operating margins & Lifetime value over next 3 & 6 month to review impact.

## Methodology
### Assumptions:
- Curve, being a distributor platform, we assume our main overarching **KPI** we are optimising for is Curve's **Total Operating Profit**.
- Given the high service fee charged, we assume that most of the sales for the brands come through Curve & thus **Curve has relatively high negotiating power**.
- **Fixed costs are not considered** in this analysis and the calculated revenues are costs and assumed to be the operating revenue and operating costs respectively.
- As all the metrics given are average values with no variance information, we **ignore seasonality in demand** and assume an even distribution of orders across all weeks. Accordingly, we also assume the **growth rate does not have fluctuations across time**.
- For the calculation of LTV, we assume **net cash flow** from both revenues & costs to be at the **end of each week**.
- As a customer is someone with atleast 1 order that week, we assume:
$$
\text{Net first week cash flow} \propto \text{Customer base in that week}
$$
- The discount rate is assumed to be 10%

### Thought Process:
For each brand,
$$
\text{Value of brand to Curve} = \text{Total present value of future net cash flow related to the brand}
$$

As the growth rate of customer base is the same as the growth rate of net cash flow at the end of each period(week), we can write:

$$
\text{Value of brand} = \sum_{t=1}^{n} \frac{\text{CF} \times (1 + g)^t}{(1 + d)^t}
$$

$$
where\
\ CF = \text{Net cash flow at end of first week},
\ g = \text{Weekly growth rate of cash flows},
\ n = \text{Number of weeks},
\ d = \text{Weekly Discount rate}
$$

As this is similar to calculating the **Present Value of a Growing Annuity**, we use the relevant formula to simplify the above:

$$
\text{Value of brand} = \text{CF} \times \frac{1 - \left(\frac{1 + g}{1 + d}\right)^n}{d - g}
$$

$$
\text{where} \
\ \text{CF} = \text{Net cash flow at end of first week},
\ g = \text{Weekly growth rate of cash flows},
\ d = \text{Weekly Discount rate},
\ n = \text{Number of weeks}
$$


**Net first month cash flow** is the **Total Operating Profit at the end of the first month**. Thus,

$$
\text{CF(Net first month cash flow)} = \text{Operating Profit per brand}
$$
$$
\text{CF} = \text{Brand service fee received} − \text{Delivery Costs}
$$

$$
\text{CF} = \text{Average Weekly Customers} \times \text{Weekly Orders Per Customer} \times \left(\text{(Average Basket Size} \times \text{Service Fee Percentage)} - \text{Average Cost per Distribution}\right)
$$

### Aim:
According to our assumptions, the **Operating Profit** and thus the **Value of Brand to Curve** is a function of **Average Weekly Customers** across time & **Service Fee Percentage**

Our goal is to optimise for the **Operating Profit**, i.e, optimise for the **Average Weekly Customers** across time

### Metrics Used:
- Average Weekly Customers
- Customer Growth Rate
- Average Basket Size
- Weekly Orders Per Customer
- Service Fee Percentage
- Average Cost per Distribution




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants for each brand
# Annual discount rate assumption and weekly equivalent
annual_discount_rate = 0.10
weekly_discount_rate = (1 + annual_discount_rate)**(1/52) - 1  # Convert annual rate to weekly

# Data for each brand in df
data = {
    "Brand": ["Brand 1", "Brand 2", "Brand 3"],
    "Average Weekly Customers": [1000, 1300, 2000],
    "Customer Growth Rate": [0.05, 0.05, 0.08],
    "Average Basket Size": [22, 18, 12],
    "Weekly Orders Per Customer": [1, 1.1, 1.3],
    "Service Fee Percentage": [0.20, 0.25, 0.27],
    "Average Cost Per Distribution": [4, 3, 3]
}

# Create df
df = pd.DataFrame(data)
```

### Data Summary:


```python
# EDA

# Perform exploratory data analysis (EDA) by calculating summary statistics
summary_stats = df.describe()

summary_stats
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
      <th>Average Weekly Customers</th>
      <th>Customer Growth Rate</th>
      <th>Average Basket Size</th>
      <th>Weekly Orders Per Customer</th>
      <th>Service Fee Percentage</th>
      <th>Average Cost Per Distribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1433.333333</td>
      <td>0.060000</td>
      <td>17.333333</td>
      <td>1.133333</td>
      <td>0.240000</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>513.160144</td>
      <td>0.017321</td>
      <td>5.033223</td>
      <td>0.152753</td>
      <td>0.036056</td>
      <td>0.577350</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.050000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>0.200000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1150.000000</td>
      <td>0.050000</td>
      <td>15.000000</td>
      <td>1.050000</td>
      <td>0.225000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1300.000000</td>
      <td>0.050000</td>
      <td>18.000000</td>
      <td>1.100000</td>
      <td>0.250000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1650.000000</td>
      <td>0.065000</td>
      <td>20.000000</td>
      <td>1.200000</td>
      <td>0.260000</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2000.000000</td>
      <td>0.080000</td>
      <td>22.000000</td>
      <td>1.300000</td>
      <td>0.270000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate correlation matrix (excluding non-numeric 'Brand' column)
correlation_matrix = df.drop(columns="Brand").corr()

correlation_matrix
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
      <th>Average Weekly Customers</th>
      <th>Customer Growth Rate</th>
      <th>Average Basket Size</th>
      <th>Weekly Orders Per Customer</th>
      <th>Service Fee Percentage</th>
      <th>Average Cost Per Distribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Average Weekly Customers</th>
      <td>1.000000</td>
      <td>0.956325</td>
      <td>-0.993735</td>
      <td>0.999322</td>
      <td>0.891783</td>
      <td>-0.731307</td>
    </tr>
    <tr>
      <th>Customer Growth Rate</th>
      <td>0.956325</td>
      <td>1.000000</td>
      <td>-0.917663</td>
      <td>0.944911</td>
      <td>0.720577</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>Average Basket Size</th>
      <td>-0.993735</td>
      <td>-0.917663</td>
      <td>1.000000</td>
      <td>-0.997176</td>
      <td>-0.936766</td>
      <td>0.802955</td>
    </tr>
    <tr>
      <th>Weekly Orders Per Customer</th>
      <td>0.999322</td>
      <td>0.944911</td>
      <td>-0.997176</td>
      <td>1.000000</td>
      <td>0.907841</td>
      <td>-0.755929</td>
    </tr>
    <tr>
      <th>Service Fee Percentage</th>
      <td>0.891783</td>
      <td>0.720577</td>
      <td>-0.936766</td>
      <td>0.907841</td>
      <td>1.000000</td>
      <td>-0.960769</td>
    </tr>
    <tr>
      <th>Average Cost Per Distribution</th>
      <td>-0.731307</td>
      <td>-0.500000</td>
      <td>0.802955</td>
      <td>-0.755929</td>
      <td>-0.960769</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Brand Analysis
### Lifetime Value of Brand Partnerships

Using the thought process outlined earlier, we can calculate the Lifetime Value of each brand.

As consumer brand partnerships with such growth rate are prone to renegotiation and 1 year is a good period to assume the status quo to remain stable.


```python
# Period (1 year in weeks)
n_weeks = 52

# Calculate weekly service fee and LTV for each brand
def calculate_weekly_service_fee(row):
    weekly_revenue_per_customer = row["Average Basket Size"] * row["Weekly Orders Per Customer"]
    weekly_service_fee = weekly_revenue_per_customer * row["Average Weekly Customers"] * row["Service Fee Percentage"]
    return weekly_service_fee

def calculate_weekly_distribution_cost(row):
    return row["Average Cost Per Distribution"] * row["Average Weekly Customers"]

def calculate_net_weekly_operating_profit(row):
    weekly_service_fee = calculate_weekly_service_fee(row)
    distribution_cost = calculate_weekly_distribution_cost(row)
    return weekly_service_fee - distribution_cost

def calculate_ltv(row):
    # Calculate the LTV for each row using the weekly growth rate and weekly discount rate
    net_weekly_operating_profit = calculate_net_weekly_operating_profit(row)
    growth_rate = row["Customer Growth Rate"]
    pv = (net_weekly_operating_profit * 
          (1 - ((1 + growth_rate) / (1 + weekly_discount_rate)) ** n_weeks) / 
          (weekly_discount_rate - growth_rate))
    return pv

# Apply the calculations to each row in the df
df["Weekly Service Fee"] = df.apply(calculate_weekly_service_fee, axis=1)
df["Weekly Distribution Cost"] = df.apply(calculate_weekly_distribution_cost, axis=1)
df["Net Weekly Operating Profit"] = df.apply(calculate_net_weekly_operating_profit, axis=1)
df["LTV of Future Operating Profit"] = df.apply(calculate_ltv, axis=1)

# Display the DataFrame with calculated LTV values
df[['Brand', 'Weekly Service Fee', 'Net Weekly Operating Profit', 'LTV of Future Operating Profit']]

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
      <th>Brand</th>
      <th>Weekly Service Fee</th>
      <th>Net Weekly Operating Profit</th>
      <th>LTV of Future Operating Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brand 1</td>
      <td>4400.0</td>
      <td>400.0</td>
      <td>8.714517e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brand 2</td>
      <td>6435.0</td>
      <td>2535.0</td>
      <td>5.522825e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brand 3</td>
      <td>8424.0</td>
      <td>2424.0</td>
      <td>1.511259e+06</td>
    </tr>
  </tbody>
</table>
</div>



**INFERENCE 1: From the above, we can infer that Brand 3 has the highest Lifetime Value for Curve.**


```python
# Create figure with better size and spacing
fig = plt.figure(figsize=(10, 12))


# LTV Comparison (Main focus)
plt.subplot(2, 2, 1)
bars = plt.bar(df['Brand'], df['LTV of Future Operating Profit'] / 1000)
plt.title('Lifetime Value by Brand (£ Thousands)', fontsize=12, pad=15)
plt.ylabel('LTV (£000s)', fontsize=10)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'£{height:,.0f}k',
             ha='center', va='bottom', fontsize=9)

```


    
![png](/assets/images/2025/2025-01-03-brand-value-analysis/output_10_0.png)
    



```python
# Weekly Operating Metrics

# Assuming df['Weekly Service Fee'] and df['Weekly Distribution Cost'] are already defined in your DataFrame
# Calculate Operating Profit Margin
df['Operating Profit Margin'] = ((df['Weekly Service Fee'] - df['Weekly Distribution Cost']) / df['Weekly Service Fee']) * 100

# Define metrics and positioning
metrics = ['Weekly Service Fee', 'Weekly Distribution Cost']
x = np.arange(len(df['Brand']))
width = 0.35  # Adjust bar width to fit both bars side by side

# Create a bar plot for Weekly Service Fee and Weekly Distribution Cost
fig, ax1 = plt.subplots()

# Plot side-by-side bars for Weekly Service Fee and Weekly Distribution Cost
bars1 = ax1.bar(x - width/2, df['Weekly Service Fee'], width, label='Weekly Service Fee', color='green')
bars2 = ax1.bar(x + width/2, df['Weekly Distribution Cost'], width, label='Weekly Distribution Cost', color='salmon')

# Add value labels to bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height, f'£{height:,.0f}', ha='center', va='bottom', fontsize=9, color='green')

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height, f'£{height:,.0f}', ha='center', va='bottom', fontsize=9, color='salmon')

# Create a second y-axis for Operating Profit Margin
ax2 = ax1.twinx()
# Use scatter instead of plot to remove dotted lines
scatter = ax2.scatter(x, df['Operating Profit Margin'], marker='o', color='black', label='Operating Profit Margin (%)')

# Add value labels for Operating Profit Margin points in black
for i, margin in enumerate(df['Operating Profit Margin']):
    ax2.text(i, margin, f'{margin:.1f}%', ha='center', va='bottom', fontsize=9, color='black')

# Chart title and labels
plt.title('Weekly Operating Metrics', fontsize=12, pad=15)
ax1.set_ylabel('Amount (£)', fontsize=10)
ax2.set_ylabel('Operating Profit Margin (%)', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Brand'])

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()  # Get legend from the first axis
lines2, labels2 = ax2.get_legend_handles_labels()  # Get legend from the second axis
ax1.legend(lines + lines2, labels + labels2, fontsize=9, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)

# Set limits for the second y-axis
ax2.set_ylim(0, max(df['Operating Profit Margin']) * 1.1)

plt.show()

```


    
![png](/assets/images/2025/2025-01-03-brand-value-analysis/output_11_0.png)
    


## Additional Considerations

### Potential Impactful Factors
- **Lack of variance metrics:** We do not have the variance for any of the metrics. This information could add value to the confidence of our observations significantly and impact the results in some cases - particularly with the growth rate.
- **Customer distribution & loyalty:** We do not know if there is a significant customer overlap between the different brands. Lack of information about customer loyalty means that we cannot know if there will be significant cannibalisation of sales over time.
- **Fixed costs:** As the only costs given are variable, we ignore fixed costs - in the real world, the fixed costs should be taken into consideration.
- **Seasonality:** Given the average data given, we have ignored seasonality completely, assuming all metrics to be usable in a time period of a year. In retail, sales tend to be seasonal with large spikes and troughs over different weeks.
- **Marketing mix uncertainty:** In this scenario, as we do not know how saturated the market is, organic vs paid customer base, new vs repeat customer base, we cannot be sure about the stability of the given growth rate in the real world.
- **Operating cycle differences:** Operating cycle for a retail distribution platform tends to be much longer than a week in terms of cash flow, i.e, 30-60 days. We have assumed weekly operating cycle for the sake of simplicity of calculation. Variations in the operating cycle across brands could potentially change the LTV of said brands although not by much.

## Service Fee Recommendations

### Assumptions:
From our earlier formula:

$$
\text{CF} = \text{Average Weekly Customers} \times \text{Weekly Orders Per Customer} \times \left(\text{(Average Basket Size} \times \text{Service Fee Percentage)} - \text{Average Cost per Distribution}\right)
$$

- We assume that each brand has a **separate customer base** and **grows independently** of each other.
- Also, assuming **customer loyalty is high** - **Average Basket Size** & **Weekly Orders Per Customer** would be **independent of service fee changes**.
- Thus **changing the service fee** would only impact **Average Weekly Customers**, adding a certain amount of churn. This can be represented as an **impact of the growth rate**.
- We assume that any change made to our service fee will directly impact the pricing of the Brand's; the final cost of item paid by customer will be variable
- Out **delivery charges** do not change across time.

We will do a scenario analysis to find the impact of varying the service fee and its impact on the Lifetime Value of each brand.


```python
def simulate_service_fee_impact(df, fee_range=np.arange(0.20, 0.31, 0.005)):
    """Simulate the impact of different service fees on profit and LTV"""
    results = []
    
    for brand in df['Brand']:
        brand_data = df[df['Brand'] == brand].iloc[0]
        
        for fee in fee_range:
            # Create a copy of the row with modified service fee
            modified_row = brand_data.copy()
            modified_row['Service Fee Percentage'] = fee
            
            # Calculate weekly metrics
            weekly_revenue = modified_row['Average Weekly Customers'] * \
                           modified_row['Weekly Orders Per Customer'] * \
                           modified_row['Average Basket Size']
            
            weekly_service_fee = weekly_revenue * fee
            
            weekly_distribution_cost = modified_row['Average Weekly Customers'] * \
                                    modified_row['Weekly Orders Per Customer'] * \
                                    modified_row['Average Cost Per Distribution']
            
            net_weekly_profit = weekly_service_fee - weekly_distribution_cost
            
            # Calculate LTV
            growth_rate = modified_row['Customer Growth Rate']
            ltv = (net_weekly_profit / (weekly_discount_rate - growth_rate)) * \
                  (1 - ((1 + growth_rate) / (1 + weekly_discount_rate)) ** n_weeks)
            
            # Calculate price sensitivity impact (simplified model)
            # Assume higher fees lead to lower growth rates
            growth_impact = 1 - ((fee - 0.20) * 2)  # Linear impact model
            adjusted_growth = growth_rate * max(0.5, growth_impact)  # Cap the minimum impact
            
            # Calculate adjusted LTV with growth impact
            adjusted_ltv = (net_weekly_profit / (weekly_discount_rate - adjusted_growth)) * \
                         (1 - ((1 + adjusted_growth) / (1 + weekly_discount_rate)) ** n_weeks)
            
            results.append({
                'Brand': brand,
                'Service Fee': fee,
                'Weekly Profit': net_weekly_profit,
                'LTV': ltv,
                'Adjusted LTV': adjusted_ltv,
                'Adjusted Growth Rate': adjusted_growth
            })
    
    return pd.DataFrame(results)

# Run simulation
simulation_results = simulate_service_fee_impact(df)

# Analysis visualization
def plot_optimization_results(simulation_results):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Weekly Profit vs Service Fee
    sns.lineplot(data=simulation_results, x='Service Fee', y='Weekly Profit', 
                hue='Brand', ax=axes[0])
    axes[0].set_title('Weekly Profit vs Service Fee')
    axes[0].set_xlabel('Service Fee Percentage')
    axes[0].set_ylabel('Weekly Profit (£)')
    
    # Plot 2: Adjusted LTV vs Service Fee
    sns.lineplot(data=simulation_results, x='Service Fee', y='Adjusted LTV', 
                hue='Brand', ax=axes[1])
    axes[1].set_title('Adjusted LTV vs Service Fee (Including Growth Impact)')
    axes[1].set_xlabel('Service Fee %')
    axes[1].set_ylabel('Adjusted LTV (£)')
    
    plt.tight_layout()
    plt.show()

# Find optimal fees
def get_optimal_fees(simulation_results):
    optimal_fees = []
    
    for brand in simulation_results['Brand'].unique():
        brand_data = simulation_results[simulation_results['Brand'] == brand]
        
        # Find fee that maximizes adjusted LTV
        optimal_fee = brand_data.loc[brand_data['Adjusted LTV'].idxmax()]
        
        current_fee = df[df['Brand'] == brand]['Service Fee Percentage'].iloc[0]
        
        # Check if there is data for the current fee in the simulation results
        current_data = brand_data[brand_data['Service Fee'] == current_fee]
        if current_data.empty:
            continue
        
        optimal_fees.append({
            'Brand': brand,
            'Current Fee': current_fee,
            'Optimal Fee': optimal_fee['Service Fee'],
            'Current Weekly Profit': current_data['Weekly Profit'].iloc[0],
            'Optimal Weekly Profit': optimal_fee['Weekly Profit'],
            'Current Adjusted LTV': current_data['Adjusted LTV'].iloc[0],
            'Optimal Adjusted LTV': optimal_fee['Adjusted LTV']
        })
    
    return pd.DataFrame(optimal_fees)

# Generate visualizations
plot_optimization_results(simulation_results)
optimization_results = get_optimal_fees(simulation_results)
```


    
![png](/assets/images/2025/2025-01-03-brand-value-analysis/output_14_0.png)
    


### Observations:
- Brand 3 has higher growth, although any service fee increase will increase the pricing by Brand accordingly, the churn will be nullified by the  up to 30 % service fee
- Brand 1 & 2 have relatively lower value from increases in Lifetime Value with increase in service fee due to the impact of low growth.

### Suggested Changes:
- Primary action is to increase Brand 3 Service Fee to 30% : As growth is high, churn will be nullified.
- Brand 1 & 2 service fee to be increased to 21% and 26% respectively to test impact
- To review all brands operating margins & Lifetime value over next 3 & 6 month to review impact.

### Expected Brand Responses:
- The expected brand response will be that they have to increase their own costs to cover the increase in service fee
- This may cause them to assume the customer base will churn uncontrollably
- To address this:
    - For Brand 3, as we are confident on minimal churn we can keep the 30% margin with our negotiating power
    - For Brands 1 & 2, as we are less confident, post negotiations, we can settle for a lesser


```python

```
