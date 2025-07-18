import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and data
pipe = joblib.load(open("pipe_laptop.pkl", "rb"))
df = joblib.load(open("df_laptop.pkl", "rb"))

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

st.title("💻 Laptop Price Predictor")
st.write("Use this app to predict laptop prices based on configuration.")


# Helper functions for default selections
def default_or_first(series):
    median = series.median()
    if median in series.unique():
        return median
    else:
        return series.mode().iloc[0]


def default_select(series):
    return series.mode().iloc[0]


brand = st.selectbox(
    "Brand",
    sorted(df["brand"].unique()),
    index=list(sorted(df["brand"].unique())).index(default_select(df["brand"])),
)

# CPU brand and name
cpu_name = st.selectbox(
    "CPU Name",
    list(sorted(df["cpu_name"].unique())) + ["other"],
    index=list(sorted(df["cpu_name"].unique())).index(default_select(df["cpu_name"])),
)

cpu_cores = st.select_slider(
    "CPU Cores",
    options=sorted(df["cpu_cores"].unique()),
    value=int(default_or_first(df["cpu_cores"])),
)
cpu_threads = st.select_slider(
    "CPU Threads",
    options=sorted(df["cpu_threads"].unique()),
    value=int(default_or_first(df["cpu_threads"])),
)

# RAM
ram_capacity = st.select_slider(
    "RAM Capacity (GB)",
    options=sorted(df["ram_capacity"].unique()),
    value=int(default_or_first(df["ram_capacity"])),
)
ram_ddr_type = st.selectbox(
    "RAM DDR Type",
    sorted(df["ram_ddr_type"].unique()),
    index=list(sorted(df["ram_ddr_type"].unique())).index(
        default_select(df["ram_ddr_type"])
    ),
)

# Storage
ssd = st.select_slider(
    "SSD Size (GB)",
    options=sorted(df["ssd"].unique()),
    value=int(default_or_first(df["ssd"])),
)
hdd = st.select_slider(
    "HDD Size (GB)",
    options=sorted(df["hdd"].unique()),
    value=int(default_or_first(df["hdd"])),
)

# GPU
gpu_name = st.selectbox("GPU Name", sorted(df["gpu_name"].unique()) + ["other"], index=0)

# Screen
screen_size = st.select_slider(
    "Screen Size (inches)",
    options=sorted(df["screen_size"].unique()),
    value=float(default_or_first(df["screen_size"])),
)
screen_res = st.selectbox(
    "Screen Resolution",
    sorted(df["screen_res"].unique()),
    index=list(sorted(df["screen_res"].unique())).index(
        default_select(df["screen_res"])
    ),
)
ppi = st.select_slider(
    "PPI", options=sorted(df["ppi"].unique()), value=float(default_or_first(df["ppi"]))
)
aspect_ratio_category = st.selectbox(
    "Aspect Ratio",
    sorted(df["aspect_ratio_category"].unique()),
    index=list(sorted(df["aspect_ratio_category"].unique())).index(
        default_select(df["aspect_ratio_category"])
    ),
)

# Touchscreen
touchscreen = st.selectbox("Touchscreen", ["Yes", "No"], index=0)
touchscreen_binary = 1 if touchscreen == "Yes" else 0

# OS
os = st.selectbox(
    "Operating System",
    sorted(df["os"].unique()),
    index=list(sorted(df["os"].unique())).index(default_select(df["os"])),
)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        {
            "brand": [brand],
            "cpu_name": [cpu_name],
            "cpu_cores": [cpu_cores],
            "cpu_threads": [cpu_threads],
            "ram_capacity": [ram_capacity],
            "ram_ddr_type": [ram_ddr_type],
            "ssd": [ssd],
            "hdd": [hdd],
            "gpu_name": [gpu_name],
            "screen_size": [screen_size],
            "screen_res": [screen_res],
            "ppi": [ppi],
            "aspect_ratio_category": [aspect_ratio_category],
            "touchscreen": [touchscreen_binary],
            "os": [os],
        }
    )

    predicted_log_price = pipe.predict(input_data)[0]
    predicted_price = np.exp(predicted_log_price)

    st.success(f"💰 **Predicted Price: ₹ {predicted_price:,.0f}**")

    # Show similar laptops in dataset
    lower_price = int(predicted_price * 0.95)
    upper_price = int(predicted_price * 1.05)
    similar_laptops = df[
        (df["brand"] == brand)
        & (df["os"] == os)
        & (df["ram_capacity"] == ram_capacity)
        & (df["ssd"] == ssd)
        & (df["price"] > lower_price)
        & (df["price"] < upper_price)
    ][["name", "price"]]
    if not similar_laptops.empty:
        st.markdown(
            f"""
        <h3>
        📋 Similar <span style='font-weight:bold; color:#0072C6;'>{brand}</span> Laptops between 
        <span style='font-weight:bold; color:#28a745;'>₹ {similar_laptops['price'].min()}</span> and 
        <span style='font-weight:bold; color:#28a745;'>₹ {similar_laptops['price'].max()}</span> 
        with <span style='font-weight:bold; color:#d63384;'>{os}</span> OS, 
        <span style='font-weight:bold; color:#ff8c00;'>{ram_capacity} GB RAM</span>, 
        <span style='font-weight:bold; color:#ff8c00;'>{ssd} GB SSD</span>
        </h3>
        """,
            unsafe_allow_html=True,
        )
        similar_laptops["link"] = similar_laptops["name"].apply(
            lambda name: f"https://www.smartprix.com/laptops/?q={name}"
        )
        similar_laptops["name"] = similar_laptops["name"].apply(
            lambda x: " ".join(x.split(" ")[:5])
        )
        st.dataframe(
            similar_laptops.reset_index(drop=True),
            column_config={"link": st.column_config.LinkColumn()},
        )
    else:
        st.write("No similar laptops found in dataset.")
