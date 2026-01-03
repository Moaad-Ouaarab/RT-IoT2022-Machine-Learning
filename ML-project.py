import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained models and encoders
model = joblib.load("artifacts/model_lgb.pkl")
label_encoders = joblib.load("artifacts/label_encoders.pkl")
target_encoder = joblib.load("artifacts/target_encoder.pkl")
selected_features = joblib.load("artifacts/selected_features.pkl")
feature_order = joblib.load("artifacts/feature_order.pkl")
default_values = joblib.load("artifacts/default_values.pkl")


def preprocess_input(df):
    """Preprocess input data: label encode, reorder"""
    df = df.copy()
    
    # Drop target column, index columns, and dropped columns from training
    dropped_cols = ['Attack_type', 'id', 'Unnamed: 0', 'bwd_URG_flag_count']
    df = df.drop(columns=dropped_cols, errors='ignore')
    
    # Select only columns in feature_order (removes any extra columns)
    df = df[feature_order]
    
    # Label encoding for categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError as e:
                raise ValueError(f"Invalid value in column '{col}': {str(e)}")
    
    # For missing categorical columns, fill with the first class encoded value
    for col in feature_order:
        if col not in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(
                    [label_encoders[col].classes_[0]]
                )[0]
            else:
                df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_order]
    
    return df

st.set_page_config(page_title="Network Attack Detection", layout="wide", initial_sidebar_state="expanded")

# Display header image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        # Try to load from URL 
        url = "https://www.cisa.gov/sites/default/files/styles/hero_large/public/2023-01/23_0127_ecd_ps-cyber_header_1600x600.png?h=e0d9a4bb&itok=tn8cHABi"
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        st.image(img)
    except:
        # Fallback if image fails to load
        st.warning("Could not load header image")

# Centered title
st.markdown("<h1 style='text-align: center;'>Network Attack Detection – LightGBM</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: gray;'>Detect network attacks using machine learning. Choose your input method below.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar info
with st.sidebar:
    st.info("Model Information")
    st.write(f"**Total Features:** {len(feature_order)}")
    st.write(f"**Important Features:** {len(selected_features)}")
    st.divider()
    st.markdown("**Note:** the initial form values represent the most frequent values in the RT_IOT2022 dataset")
    st.markdown("**How to use:**")
    st.markdown("""
    1. **Full feature input**: Enter all network features manually
    2. **Important features only**: Quick prediction with key features
    3. **Upload CSV**: Batch predictions for multiple records
    """)

mode = st.radio(
    "Choose input mode:",
    ["Full feature input", "Important features only", "Upload CSV"],
    help="Select how you want to provide data for predictions"
)

# ============ MODE 1: Full Feature Input ============
if mode == "Full feature input":
    st.markdown("<h3 style='text-align: center;'>Enter All Features</h3>", unsafe_allow_html=True)
    input_data = {}
    
    with st.form("full_form"):
        # Create two columns for better UI
        col1, col2 = st.columns(2)
        
        for idx, col in enumerate(feature_order):
            # Alternate between columns
            target_col = col1 if idx % 2 == 0 else col2
            
            with target_col:
                if col in label_encoders:
                    # Dropdown for categorical columns
                    options = label_encoders[col].classes_.tolist()
                    input_data[col] = st.selectbox(
                        col,
                        options,
                        key=f"full_{col}"
                    )
                else:
                    # Number input for numeric columns
                    default_val = float(default_values.get(col, 0))

                    input_data[col] = st.number_input(
    					col,
    					value=default_val,
    					key=f"full_{col}"
					)
        
        submit = st.form_submit_button("Predict", use_container_width=True)
    
    if submit:
        try:
            df = pd.DataFrame([input_data])
            X = preprocess_input(df)
            pred = model.predict(X)
            pred_proba = model.predict_proba(X)
            label = target_encoder.inverse_transform(pred)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Prediction:** {label[0]}")
            with col2:
                max_prob = pred_proba.max()
                st.info(f"**Confidence:** {max_prob:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# ============ MODE 2: Important Features Only ============
elif mode == "Important features only":
    st.markdown(f"<h3 style='text-align: center;'>Enter Important Features Only</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Provide values for {len(selected_features)} most important features</p>", unsafe_allow_html=True)
    input_data = {}
    
    with st.form("important_form"):
        col1, col2 = st.columns(2)
        
        for idx, col in enumerate(selected_features):
            target_col = col1 if idx % 2 == 0 else col2
            
            with target_col:
                if col in label_encoders:
                    # Dropdown for categorical columns
                    options = label_encoders[col].classes_.tolist()
                    input_data[col] = st.selectbox(
                        col,
                        options,
                        key=f"important_{col}"
                    )
                else:
                    # Number input for numeric columns
                    default_val = float(default_values.get(col, 0))

                    input_data[col] = st.number_input(
    					col,
    					value=default_val,
    					key=f"full_{col}"
					)
        
        submit = st.form_submit_button("Predict", use_container_width=True)
    
    if submit:
        try:
            df = pd.DataFrame([input_data])
            
            # Fill missing features with defaults
            for col in feature_order:
                if col not in df.columns:
                    if col in label_encoders:
                        # Use first class for categorical columns
                        df[col] = label_encoders[col].classes_[0]
                    else:
                        # Use 0 for numeric columns
                        df[col] = default_values[col]
            
            X = preprocess_input(df)
            pred = model.predict(X)
            pred_proba = model.predict_proba(X)
            label = target_encoder.inverse_transform(pred)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Prediction:** {label[0]}")
            with col2:
                max_prob = pred_proba.max()
                st.info(f"**Confidence:** {max_prob:.2%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# ============ MODE 3: Upload CSV ============
elif mode == "Upload CSV":
    st.markdown("<h3 style='text-align: center;'>Upload CSV File for Batch Prediction</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a CSV file with network traffic data. The file should contain the same columns as the training data.</p>", unsafe_allow_html=True)
    
    file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if file:
        try:
            # Read CSV
            df = pd.read_csv(file, index_col=0) if file.name.endswith('.csv') else pd.read_csv(file)
            
            # Drop unnamed index columns if present
            df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
            
            st.write(f"File loaded successfully. Shape: {df.shape}")
            
            # Show sample
            with st.expander("Preview data"):
                st.dataframe(df.head())
            
            # Preprocess
            with st.spinner("Processing and making predictions..."):
                # Keep original data for output
                df_original = df.copy()
                X = preprocess_input(df)
                preds = model.predict(X)
                pred_proba = model.predict_proba(X)
                labels = target_encoder.inverse_transform(preds)
                
                # Add predictions to original dataframe
                df_original["Prediction"] = labels
                df_original["Confidence"] = pred_proba.max(axis=1)
            
            # Display results
            st.success("Predictions completed!")
            st.dataframe(df_original, use_container_width=True)
            
            # Download results
            csv_data = df_original.to_csv(index=False)
            st.download_button(
                label="Download predictions",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Summary statistics
            with st.expander("Prediction Summary"):
                st.write(f"**Total predictions:** {len(labels)}")
                pred_counts = pd.Series(labels).value_counts()
                st.write(pred_counts)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Make sure your CSV has the correct columns and format.")

# Footer
st.divider()
st.markdown("<p style='text-align: center;'>Network Attack Detection System | Powered by LightGBM</p>", unsafe_allow_html=True)