# E-commerce-Analytics-Customer-Segmentation-Demand-Forecasting-Inventory-Optimization
A direct application and insight finding data science project with end-to-end e-commerce data for fast-moving consumer goods.

## 1. Project Overview

This is an end-to-end e-commerce data science project simulating the analytics workflow of a Direct-to-Consumer (D2C) Fast-Moving Consumer Goods (FMCG) retailer. In the competitive D2C market, businesses face two core challenges:

1.  **Customer Heterogeneity:** Unlike traditional B2B models with a few large clients, D2C involves a massive, diverse, and rapidly changing customer base.
2.  **Supply Chain Strain:** D2C orders are characterized by "small batch, high frequency," placing significant pressure on demand forecasting and inventory management, which directly impacts costs and customer satisfaction.

This project leverages machine learning to address three critical business problems, starting from raw transactional data.

## 2. Business Objectives

1.  **Customer Insight:** Who are our customers? What is their value? (**Customer Segmentation**)
2.  **Demand Forecasting:** What will the sales volume be for our key products? (**Time Series Forecasting**)
3.  **Cost Optimization:** How can we manage inventory to prevent stockouts and overstocking, thereby saving costs? (**Supply Chain Optimization**)

## 3. Dataset

* **Name:** Online Retail Dataset
* **Source:** UCI Machine Learning Repository
* **Description:** Contains all transactions for a UK-based online retailer between December 2010 and December 2011.
* **Key Fields:**
    * `InvoiceNo`: Invoice number
    * `StockCode`: Product (item) code
    * `Description`: Product name
    * `Quantity`: The quantities of each product per transaction
    * `InvoiceDate`: The day and time when each transaction was generated
    * `UnitPrice`: Product price per unit
    * `CustomerID`: Customer number
    * `Country`: The name of the country where each customer resides

## 4. Tech Stack

* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning (ML):**
    * **Clustering:** Scikit-learn (KMeans, StandardScaler)
    * **Forecasting:** Prophet (Facebook)
* **Environment:** Jupyter Notebook

## 5. Project Pipeline & Methodology

This project is divided into three core modules, corresponding to three Jupyter Notebooks.

### Module 1: Data Cleaning and Exploratory Data Analysis (EDA)

* **Objective:** To process the raw data into a clean, analysis-ready dataset.
* **Pipeline:**
    1.  Load data, handling encoding issues.
    2.  Handle missing values (especially `CustomerID`).
    3.  Clean anomalous data (e.g., negative `Quantity` values, which represent returns).
    4.  Feature Engineering: Create the `TotalPrice` feature.
    5.  EDA: Analyze overall sales trends, top 10 best-selling products, and customer country distribution.

### Module 2: Machine Learning Methods for Customer Segmentation (RFM + K-Means)

* **Objective:** To identify distinct groups of customers to enable targeted marketing.
* **Methodology:**
    1.  **RFM Model:** Extract three key metrics for each customer:
        * **R (Recency):** Days since the last purchase (measures engagement).
        * **F (Frequency):** Total number of transactions (measures loyalty).
        * **M (Monetary):** Total amount spent (measures value).
    2.  **Data Preprocessing:** Apply a `log` transform to the highly-skewed RFM data, followed by `StandardScaler`.
    3.  **K-Means Clustering:**
        * Use the "Elbow Method" to determine the optimal number of clusters (K).
        * Train the K-Means model and assign a cluster label to each customer.
    4.  **Segment Analysis:** Analyze the RFM mean values for each cluster to define customer personas (e.g., "Champions," "At-Risk," "New Customers") and propose marketing strategies.

### Module 3: Machine Learning Methods for Demand Forecasting & Inventory Optimization (Prophet + EOQ)

* **Objective:** To forecast future demand for a core SKU and optimize inventory policy to save costs.
* **Methodology (Part 1 - Forecasting):**
    1.  **Data Preparation:** Select a high-volume product and aggregate its transactions into a daily sales time series.
    2.  **Model Training:** Use Facebook's `Prophet` model. Prophet excels at handling time series with multiple seasonalities (weekly, yearly) and holiday effects.
    3.  **Generate Forecast:** Predict the next 30 days of sales, including uncertainty intervals.

* **Methodology (Part 2 - Optimization):**
    1.  **Define Parameters:** Based on business assumptions, define inventory parameters (e.g., lead time, order cost, holding cost).
    2.  **Safety Stock:** Use the standard deviation from Prophet's forecast to calculate the buffer stock needed to hedge against demand volatility.
    3.  **Reorder Point (ROP):** Calculate "when" to place a new order.
        * `ROP = (Average Daily Demand × Lead Time) + Safety Stock`
    4.  **Economic Order Quantity (EOQ):** Calculate "how much" to order to minimize total inventory costs (ordering + holding).
        * $EOQ = \sqrt{\frac{2 \times Annual_Demand \times Order_Cost}{Annual_Holding_Cost_per_Unit}}$

## 6. Core Insights & Business Value

* **Value 1 (Targeted Marketing):** Successfully segmented customers into 4 distinct groups. This allows the marketing team to move from "shotgun" campaigns to targeted actions: sending re-engagement emails to "At-Risk" customers and offering VIP perks to "Champions," thus optimizing the marketing budget.
* **Value 2 (Supply Chain Insight):** For core SKU 'XYZ', we forecast an average daily demand of `Y` units for the next 30 days.
* **Value 3 (Cost Savings):** Based on the forecast, we calculated the Reorder Point (ROP) as `A` units and the Economic Order Quantity (EOQ) as `B` units. This means:
    * When inventory drops to `A` units, a purchase order is automatically triggered.
    * Ordering `B` units at a time minimizes total inventory costs.
    * This model can **save an estimated XX% in warehousing costs** and **reduce stockout rates by YY%** annually.
