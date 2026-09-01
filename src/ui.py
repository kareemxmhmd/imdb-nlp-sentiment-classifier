import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from io import StringIO

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ReviewPulse", layout="wide")

st.title("ReviewPulse: Sentiment Classification & Triage")

# Navigation
page = st.sidebar.selectbox("Navigate", ["Single Review Tester", "Batch Upload", "Trends Dashboard", "Review Queue"])

# Model Info in sidebar
st.sidebar.markdown("---")
try:
    info_res = requests.get(f"{API_URL}/model/info")
    if info_res.status_code == 200:
        info = info_res.json()
        st.sidebar.subheader("Model Info")
        st.sidebar.text(f"Version: {info['version']}")
        st.sidebar.text(f"Accuracy: {info['accuracy']:.2%}")
        st.sidebar.text(f"Auto-Tag Threshold: {info['auto_tag_threshold']:.2f}")
    else:
        st.sidebar.warning("API not reachable or model not loaded.")
except requests.exceptions.ConnectionError:
    st.sidebar.error("Could not connect to API. Is it running?")


if page == "Single Review Tester":
    st.header("Single Review Tester")
    st.write("Test the sentiment classification and decision triage live.")
    
    review_text = st.text_area("Enter a review:", height=150)
    
    if st.button("Predict"):
        if review_text:
            try:
                res = requests.post(f"{API_URL}/predict", json={"text": review_text})
                if res.status_code == 200:
                    data = res.json()
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_color = "green" if data['sentiment'] == "Positive" else "red"
                    col1.metric("Sentiment", data['sentiment'])
                    col2.metric("Confidence", f"{data['confidence']:.2f}")
                    
                    tier_color = "orange" if "Human" in data['decision_tier'] else "blue"
                    col3.metric("Decision Tier", data['decision_tier'])
                    
                    st.info(f"Reason: {data['reason']}")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
        else:
            st.warning("Please enter some text.")

elif page == "Batch Upload":
    st.header("Batch Upload")
    st.write("Upload a CSV with a 'review' column to process in batch.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            st.write(f"Loaded {len(df)} reviews. Preview:")
            st.dataframe(df.head())
            
            if st.button("Process Batch"):
                reviews = df['review'].tolist()
                
                with st.spinner('Processing...'):
                    try:
                        res = requests.post(f"{API_URL}/predict/batch", json={"reviews": reviews})
                        if res.status_code == 200:
                            results = res.json()['results']
                            res_df = pd.DataFrame(results)
                            st.success("Batch processed successfully!")
                            st.dataframe(res_df)
                            
                            csv = res_df.to_csv(index=False)
                            st.download_button("Download Results", csv, "batch_results.csv", "text/csv")
                        else:
                            st.error(f"API Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

elif page == "Trends Dashboard":
    st.header("Trends Dashboard")
    st.write("Simulated sentiment trends over time by product.")
    
    try:
        res = requests.get(f"{API_URL}/trends")
        if res.status_code == 200:
            trends = res.json()
            df = pd.DataFrame(trends)
            
            # Check for alerts
            alerts = df[df['alert_triggered'] == True]
            if not alerts.empty:
                for _, row in alerts.iterrows():
                    st.error(f"⚠️ SPIKE ALERT: High negative sentiment for '{row['product_title']}' on {row['timestamp']} (Neg rate: {row['negative_rate']:.1%})")
            
            # Plot
            fig = px.line(df, x="timestamp", y="negative_rate", color="product_title", 
                          title="Negative Sentiment Rate Over Time",
                          markers=True,
                          labels={"negative_rate": "Negative Review %", "timestamp": "Date", "product_title": "Product"})
            
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Raw Trend Data")
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Could not load trends: {e}")

elif page == "Review Queue":
    st.header("Review Queue (Human-in-the-Loop)")
    st.write("Reviews flagged as 'Needs Human Review' would appear here.")
    
    # Mock data for demonstration
    st.info("Showing mock data for demonstration purposes.")
    mock_data = [
        {"id": 1, "text": "It was okay, not great but not terrible either. Kinda long.", "confidence": 0.52},
        {"id": 2, "text": "The acting was superb, however the plot felt very disjointed and left me confused.", "confidence": 0.65},
        {"id": 3, "text": "Some good parts, some bad parts. Needs more explosions.", "confidence": 0.58}
    ]
    
    for item in mock_data:
        with st.expander(f"Review ID {item['id']} (Confidence: {item['confidence']})"):
            st.write(item['text'])
            col1, col2, col3 = st.columns(3)
            if col1.button("Mark Positive", key=f"pos_{item['id']}"):
                st.success(f"Review {item['id']} marked as Positive.")
            if col2.button("Mark Negative", key=f"neg_{item['id']}"):
                st.success(f"Review {item['id']} marked as Negative.")
            if col3.button("Discard", key=f"dis_{item['id']}"):
                st.warning(f"Review {item['id']} discarded.")
