import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Page settings
st.set_page_config(page_title="E-Commerce Analytics", layout="wide")

# Title
st.title("🛒 E-Commerce Customer Behavior Analysis")

# Load dataset
df = pd.read_csv("customer_data.csv")

# Show dataset
st.subheader("Customer Dataset")
st.dataframe(df)

# Metrics
total_sales = df["Purchase_Amount"].sum()
avg_rating = df["Rating"].mean()

col1, col2 = st.columns(2)

col1.metric("Total Sales", f"₹{total_sales}")
col2.metric("Average Rating", round(avg_rating, 2))

# Product category chart
st.subheader("Top Product Categories")

fig, ax = plt.subplots(figsize=(8,4))
sns.countplot(x='Product_Category', data=df, ax=ax)
plt.xticks(rotation=45)

st.pyplot(fig)

# Payment method pie chart
st.subheader("Payment Method Distribution")

fig2, ax2 = plt.subplots()

df['Payment_Method'].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=ax2
)

st.pyplot(fig2)

# Purchase amount by gender
st.subheader("Purchase Amount by Gender")

fig3, ax3 = plt.subplots()

sns.barplot(
    x='Gender',
    y='Purchase_Amount',
    data=df,
    ax=ax3
)

st.pyplot(fig3)

# Customer Segmentation
st.subheader("Customer Segmentation")

X = df[['Age', 'Purchase_Amount']]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

fig4, ax4 = plt.subplots()

sns.scatterplot(
    x=df['Age'],
    y=df['Purchase_Amount'],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)

plt.xlabel("Age")
plt.ylabel("Purchase Amount")

st.pyplot(fig4)

# Insights
st.subheader("Business Insights")

top_category = df['Product_Category'].value_counts().idxmax()
top_payment = df['Payment_Method'].value_counts().idxmax()

st.success(f"Top Selling Category: {top_category}")
st.success(f"Most Used Payment Method: {top_payment}")

st.info("Customer segmentation helps businesses target users effectively.")