# E-commerce Analytics: Customer Segmentation, Demand Forecasting & Inventory Optimization

This is an end-to-end data science project demonstrating how to derive actionable business insights from raw e-commerce data. The project simulates the workflow for a Direct-to-Consumer (D2C) retailer, moving from data cleaning and EDA to advanced machine learning for customer segmentation and supply chain optimization.

## 1. Project Overview

In the highly competitive D2C (Direct-to-Consumer) market, businesses can no longer rely on intuition. This project tackles two of the primary challenges D2C retailers face:

1.  **Customer Heterogeneity:** Unlike traditional B2B models with a few large clients, D2C involves a massive, diverse, and rapidly changing customer base. Understanding "who" the customers are is key to retention and marketing ROI.
2.  **Supply Chain Strain:** D2C orders are characterized by "small batch, high frequency," which places immense pressure on inventory. Overstocking burns cash, while understocking (stockouts) leads to lost sales and customer churn.

This project leverages machine learning to provide data-driven solutions to both challenges, creating a framework for intelligent e-commerce operations.

## 2. Business Objectives

1.  **Customer Insight:** Move from an unknown, monolithic customer base to clear, actionable segments. Answer: "Who are our customers, and what is their value?"
2.  **Demand Forecasting:** Move from reactive to predictive ordering. Answer: "What will the sales be for our key products?"
3.  **Cost Optimization:** Use the forecast to create an automated, cost-saving inventory policy. Answer: "How can we minimize inventory costs while maximizing product availability?"

## 3. Dataset

* **Name:** Online Retail Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Description:** A transnational dataset containing all transactions for a UK-based online retailer between 12/01/2010 and 12/09/2011.
* **Key Fields:**
    * `InvoiceNo`: Invoice number.
    * `StockCode`: Product (item) code.
    * `Description`: Product name.
    * `Quantity`: The quantities of each product per transaction.
    * `InvoiceDate`: Transaction date and time.
    * `UnitPrice`: Product price per unit.
    * `CustomerID`: Customer number.
    * `Country`: Country where the customer resides.

## 4. Tech Stack

* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning (ML):**
    * **Clustering:** Scikit-learn (KMeans, StandardScaler)
    * **Forecasting:** Prophet (Facebook)
* **Environment:** Jupyter Notebook / VS Code

## 5. Project Pipeline & Methodology

The project is broken into three sequential modules, each in its own notebook.

### Module 1: Data Cleaning and Exploratory Data Analysis (EDA)

* **Objective:** To transform the raw, noisy dataset into a clean, reliable, and analysis-ready format.
* **Process:**
    1.  **Load Data:** Read the `.csv` file, handling the `ISO-8859-1` encoding.
    2.  **Handle Missing Data:** `CustomerID` is essential for segmentation. All rows with missing `CustomerID`s (approx. 25% of the dataset) were dropped.
    3.  **Clean Anomalies:**
        * Removed transactions with a negative `Quantity`, as these represent returns and would skew sales analysis.
        * Removed transactions with a `UnitPrice` of 0.
    4.  **Feature Engineering:** Created the `TotalPrice` (`Quantity` * `UnitPrice`) column, which is the primary metric for customer value and sales.
    5.  **EDA:** Performed exploratory analysis to understand the data's structure, focusing on:
        * **Geographic Distribution:** Confirmed the dataset is overwhelmingly UK-based, validating a single-country focus.
        * **Top-Selling SKUs:** Identified top products by quantity sold to select a viable candidate for demand forecasting.
        * **Monthly Sales Trends:** Plotted sales over time, which revealed a powerful yearly seasonality with a massive peak in Q4 (Oct-Nov).

### Module 2: Machine Learning for Customer Segmentation (RFM + K-Means)

* **Objective:** To segment all customers into distinct, actionable groups.
* **Process:**
    1.  **RFM Feature Engineering:** Aggregated data to the `CustomerID` level to create three powerful behavioral features:
        * **`Recency` (R):** Days since the customer's last purchase (calculated from a fixed "snapshot date").
        * **`Frequency` (F):** Total number of unique transactions (invoices) from the customer.
        * **`Monetary` (M):** Total sum of `TotalPrice` for the customer.
    2.  **Data Preprocessing:** The RFM features were highly right-skewed. This skew violates the assumptions of K-Means.
        * Applied a **log-transform (`np.log1p`)** to normalize the distributions.
        * Used `StandardScaler` to scale all three features to a mean of 0 and a standard deviation of 1, ensuring each feature had equal weight in the clustering algorithm.
    3.  **Model Training (K-Means):**
        * Used the **Elbow Method** to find the optimal number of clusters. By plotting the Within-Cluster Sum of Squares (WCSS) against `K` (from 1 to 10), the "elbow" point was clearly identified at **K=4**.
        * Trained a `KMeans` model with `n_clusters=4`.
    4.  **Segment Analysis:** Assigned each customer to one of the 4 clusters. Analyzed the *mean RFM values* for each cluster to build data-driven business personas.

### Module 3: ML for Demand Forecasting & Inventory Optimization (Prophet + EOQ)

* **Objective:** To forecast demand for a key product and use that forecast to build a cost-saving, automated inventory policy.
* **Process:**
    1.  **Data Preparation:**
        * Selected a top-selling, non-seasonal SKU: **'WHITE HANGING HEART T-LIGHT HOLDER' (SKU 85123A)**.
        * Aggregated all sales for this SKU into a **daily time series** (sum of `Quantity`).
        * Filled missing dates with 0 to create the continuous daily series required by the `Prophet` model.
    2.  **Forecasting (Prophet):**
        * Trained a `Prophet` model, which is ideal for this use case as it automatically detects and models `weekly_seasonality` and `yearly_seasonality`.
        * Generated a 30-day future forecast, which produces the mean prediction (`yhat`) and the uncertainty interval (`yhat_lower`, `yhat_upper`).
    3.  **Inventory Optimization (Policy Creation):**
        * Used the forecast outputs and business assumptions (e.g., lead time, order cost) to calculate three critical inventory metrics:
        * **Safety Stock (SS):** The buffer stock needed to prevent stockouts. Calculated using the `Z-score` for our 95% desired service level and the standard deviation of demand *during the lead time*.
        * **Reorder Point (ROP):** The inventory level that **triggers** a new order.
            > `ROP = (Average Daily Demand × Lead Time) + Safety Stock`
        * **Economic Order Quantity (EOQ):** The optimal order size that **minimizes** total inventory costs (balancing ordering costs vs. holding costs).
            > $EOQ = \sqrt{\frac{2 \times \text{Annual Demand} \times \text{Order Cost}}{\text{Annual Holding Cost}}}$

## 6. Core Insights & Business Value

This project successfully transitioned from raw data to actionable, data-driven strategies. The machine learning models provided precise answers to our core business objectives, moving the company from a "gut-feel" operational model to one of intelligent, optimized decision-making.

### Value 1 (Targeted Marketing): From "Shotgun" to "Surgical" Customer Segmentation

The RFM and K-Means clustering (using an optimal **K=4** clusters) successfully identified four distinct and actionable customer personas. This allows the marketing team to stop "shotgun" campaigns and deploy surgical, high-ROI strategies for each group.

Based on the cluster analysis, the personas are:

* **Cluster 1: "Champions" (Count: 1,363)**
    * **Profile:** (Low Recency ~16 days, High Frequency ~10.5 orders, High Monetary ~$2,598).
    * **Insight:** This is our most valuable group. They are active, loyal, and spend the most. They represent the core 1/3 of the customer base and are the primary revenue drivers.
    * **Strategic Action:** **Retain & Reward.** Do *not* waste margin by sending discounts. Instead, focus on building an exclusive community:
        * Implement a tiered loyalty program.
        * Grant early access to new products.
        * Provide white-glove customer service.

* **Cluster 2: "Potential Loyalists" (Count: 1,283)**
    * **Profile:** (Mid Recency ~52 days, Mid Frequency ~3.7 orders, Mid Monetary ~$724).
    * **Insight:** This group is our "growth engine." They are active and valuable but have not yet fully converted into Champions. They last purchased about 2-3 months ago.
    * **Strategic Action:** **Nurture & Upsell.**
        * Launch personalized email campaigns based on their past purchases (cross-selling).
        * Implement a points-based incentive to encourage their next purchase.

* **Cluster 3: "New Customers" (Count: 1,108)**
    * **Profile:** (Low Recency ~19 days, Low Frequency ~1.7 orders, Low Monetary ~$318).
    * **Insight:** These are recent, first-time, or second-time buyers. They are highly engaged *right now*, but their future loyalty is not guaranteed.
    * **Strategic Action:** **Onboard & Impress.**
        * Trigger an automated, multi-step welcome email series.
        * Provide value-add content (e.g., "How-to" guides).
        * Offer a compelling, one-time incentive for their *second* purchase.

* **Cluster 0: "At-Risk / Hibernating" (Count: 604)**
    * **Profile:** (High Recency ~167 days, Low Frequency ~2.0 orders, Low Monetary ~$317).
    * **Insight:** This group is dormant. They have not purchased in nearly 6 months. They are low-value and have likely churned.
    * **Strategic Action:** **Re-activate or Release.**
        * Attempt one last-ditch, high-discount "We Miss You!" campaign.
        * If they don't respond, suppress them from active marketing lists to save costs.

---

### Value 2 (Supply Chain Insight): Multi-Dimensional Demand Forecasting

The Prophet forecasting model for our top-selling SKU, **'WHITE HANGING HEART T-LIGHT HOLDER' (SKU 85123A)**, decoded the *behavior* of our demand.

* **Headline Forecast:** The model predicts an average daily demand of **106.68 units** for the next 30 days.
* **Critical Insight 1 (Yearly Seasonality):** The model confirms a *massive* and predictable demand spike in Q4, peaking in November.
    * **Business Impact:** This proves that production and inventory build-up must begin no later than Q3 (August/September) to avoid catastrophic holiday stockouts.
* **Critical Insight 2 (Weekly Seasonality):** The component plot shows that demand consistently peaks on **Thursdays and Fridays**.
    * **Business Impact:** This suggests customers are buying for weekend use.
    * **Marketing Action:** Shift promotional emails from Tuesday to Wednesday evening or Thursday morning to capture customers during their peak planning phase.
    * **Operations Action:** Ensure the warehouse is fully staffed on Fridays and Saturdays to maintain shipping speed promises.

---

### Value 3 (Cost Savings): A Data-Driven Inventory Policy

The final module translates the demand forecast into a concrete, cost-saving inventory policy. This system replaces "gut-feel" ordering with a mathematical model that optimizes cash flow and service levels.

Based on the forecast (avg. demand ~107 units, std. dev. ~59 units) and our business assumptions (7-day lead time, 95% service level, $50 order cost, $1 holding cost), we have established a new inventory doctrine:

1.  **Safety Stock: 258 units**
    * **Insight:** This is our data-driven insurance policy. It is the *precise* amount of inventory ($258 in capital) we must hold to protect against 95% of demand volatility during the 7-day lead time.

2.  **Reorder Point (ROP): 1,004 units**
    * **Insight:** This is our **trigger**. When physical inventory on-hand for SKU 85123A drops to **1,004 units**, an automated purchase order must be placed.
    * *Calculation: (Avg. Daily Demand × Lead Time) + Safety Stock = (106.7 × 7) + 258 = 1,004 units.*

3.  **Economic Order Quantity (EOQ): 1,979 units**
    * **Insight:** This is our **action**. When the ROP trigger is hit, the purchase order should be for **1,979 units**.
    * **Business Impact:** This specific quantity is the "sweet spot" that perfectly balances our two main competing costs: the *cost of ordering* (admin/shipping fees) and the *cost of holding* (warehousing, capital tied up). Ordering more or less than this amount will, by definition, be more expensive.

**Overall Value:** This project provides a framework for moving from a reactive to a predictive e-commerce business. It establishes a policy that **simultaneously minimizes warehousing costs (by preventing over-ordering)** and **prevents lost sales (by scientifically buffering against stockouts)**, directly improving the bottom line.

## 7. How to Use This Repository

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install pandas numpy matplotlib seaborn scikit-learn prophet
    ```
3.  **Download the data:**
    * Download the `Online_Retail.csv` file from the [UCI link](https://archive.ics.uci.edu/dataset/352/online+retail) and place it in the `data/` directory (you may need to create this directory).
4.  **Run the notebooks:**
    * Run the notebooks in order:
        1.  `01_Data_Cleaning_and_EDA.ipynb`
        2.  `02_Customer_Segmentation.ipynb`
        3.  `03_Demand_Forecasting_and_Inventory.ipynb`
