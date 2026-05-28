import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_and_preprocess
from src.model_training import train_model
from sklearn.metrics import accuracy_score

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airline Satisfaction Analyzer",
    page_icon="✈️",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('outputs/models/decision_tree.pkl')
    return model

@st.cache_data
def load_data():
    X_train, X_test, y_train, y_test, df = load_and_preprocess('data/train.csv')
    return X_train, X_test, y_train, y_test, df

model          = load_model()
X_train, X_test, y_train, y_test, df = load_data()

# ── Sidebar Navigation ───────────────────────────────────────
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🔍 Predict Satisfaction",
    "📊 Dashboard",
    "📈 Model Performance",
    "💡 Insights & Recommendations"
])

# ════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("✈️ Airline Customer Experience Analyzer")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Records",  f"{len(df):,}")
    col2.metric("✅ Satisfied",
                f"{(df['satisfaction']==1).sum():,}")
    col3.metric("❌ Dissatisfied",
                f"{(df['satisfaction']==0).sum():,}")
    acc = accuracy_score(y_test, model.predict(X_test))
    col4.metric("🎯 Model Accuracy", f"{acc*100:.1f}%")

    st.markdown("---")
    st.subheader("📌 About This Project")
    st.write("""
    This project uses a **Decision Tree Classifier** to predict
    whether a passenger is **Satisfied or Dissatisfied** based on:
    - Flight details (distance, delays, class)
    - Service ratings (wifi, food, entertainment, seat comfort)
    - Passenger profile (age, gender, customer type)
    """)

    st.subheader("🗂️ Dataset Sample")
    st.dataframe(df.head(10))


# ════════════════════════════════════════════════════════════
# PAGE 2 – PREDICT
# ════════════════════════════════════════════════════════════
elif page == "🔍 Predict Satisfaction":
    st.title("🔍 Predict Passenger Satisfaction")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Passenger Info")
        gender          = st.selectbox("Gender", ["Male", "Female"])
        age             = st.slider("Age", 10, 80, 30)
        customer_type   = st.selectbox("Customer Type",
                                       ["Loyal Customer", "Disloyal Customer"])
        travel_type     = st.selectbox("Type of Travel",
                                       ["Business travel", "Personal Travel"])
        flight_class    = st.selectbox("Class",
                                       ["Business", "Eco", "Eco Plus"])

    with col2:
        st.subheader("✈️ Flight Details")
        flight_distance  = st.slider("Flight Distance (km)", 100, 5000, 1000)
        departure_delay  = st.slider("Departure Delay (min)", 0, 300, 0)
        arrival_delay    = st.slider("Arrival Delay (min)", 0, 300, 0)
        wifi             = st.slider("Inflight Wifi (1-5)", 1, 5, 3)
        online_boarding  = st.slider("Online Boarding (1-5)", 1, 5, 3)

    with col3:
        st.subheader("⭐ Service Ratings")
        seat_comfort    = st.slider("Seat Comfort (1-5)", 1, 5, 3)
        food_drink      = st.slider("Food & Drink (1-5)", 1, 5, 3)
        entertainment   = st.slider("Inflight Entertainment (1-5)", 1, 5, 3)
        onboard_service = st.slider("On-board Service (1-5)", 1, 5, 3)
        cleanliness     = st.slider("Cleanliness (1-5)", 1, 5, 3)
        legroom         = st.slider("Leg Room (1-5)", 1, 5, 3)
        baggage         = st.slider("Baggage Handling (1-5)", 1, 5, 3)
        checkin         = st.slider("Check-in Service (1-5)", 1, 5, 3)
        gate_location   = st.slider("Gate Location (1-5)", 1, 5, 3)
        departure_time  = st.slider("Departure Time Convenience (1-5)", 1, 5, 3)
        online_booking  = st.slider("Ease of Online Booking (1-5)", 1, 5, 3)
        inflight_service= st.slider("Inflight Service (1-5)", 1, 5, 3)

    st.markdown("---")

    if st.button("🚀 Predict Now", use_container_width=True):
        # Encode inputs same way as training
        gender_enc       = 1 if gender == "Male" else 0
        cust_enc         = 1 if "Loyal" in customer_type else 0
        travel_enc       = 0 if "Business" in travel_type else 1
        class_enc        = 0 if flight_class == "Business" else \
                           1 if flight_class == "Eco" else 2

        features = np.array([[
            gender_enc, age, cust_enc, travel_enc, class_enc,
            flight_distance, wifi, departure_time, online_booking,
            gate_location, food_drink, online_boarding, seat_comfort,
            entertainment, onboard_service, legroom, baggage,
            checkin, inflight_service, cleanliness,
            departure_delay, arrival_delay
        ]])

        prediction   = model.predict(features)[0]
        probability  = model.predict_proba(features)[0]
        confidence   = max(probability) * 100

        if prediction == 1:
            st.success(f"✅ Passenger is **SATISFIED**  "
                       f"(Confidence: {confidence:.1f}%)")
        else:
            st.error(f"❌ Passenger is **DISSATISFIED**  "
                     f"(Confidence: {confidence:.1f}%)")

        # Confidence bar
        st.progress(int(confidence))


# ════════════════════════════════════════════════════════════
# PAGE 3 – DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Data Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Satisfaction Distribution")
        fig, ax = plt.subplots()
        df['satisfaction'].value_counts().plot(
            kind='bar', color=['#FF6B6B','#4ECDC4'], ax=ax)
        ax.set_xticklabels(['Dissatisfied','Satisfied'], rotation=0)
        st.pyplot(fig)

    with col2:
        st.subheader("Satisfaction by Travel Type")
        fig, ax = plt.subplots()
        sns.countplot(x='Type of Travel', hue='satisfaction',
                      data=df, palette=['#FF6B6B','#4ECDC4'], ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Satisfaction by Class")
        fig, ax = plt.subplots()
        sns.countplot(x='Class', hue='satisfaction',
                      data=df, palette=['#FF6B6B','#4ECDC4'], ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Delay vs Satisfaction")
        fig, ax = plt.subplots()
        sns.boxplot(x='satisfaction',
                    y='Departure Delay in Minutes',
                    data=df,
                    palette=['#FF6B6B','#4ECDC4'], ax=ax)
        ax.set_xticklabels(['Dissatisfied','Satisfied'])
        st.pyplot(fig)

    st.subheader("Service Ratings Heatmap")
    service_cols = [
        'Inflight wifi service','Seat comfort',
        'Food and drink','Inflight entertainment',
        'On-board service','Cleanliness'
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df[service_cols].corr(), annot=True,
                cmap='coolwarm', ax=ax)
    st.pyplot(fig)


# ════════════════════════════════════════════════════════════
# PAGE 4 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")

    from sklearn.metrics import (confusion_matrix,
                                  classification_report,
                                  accuracy_score)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy",  f"{acc*100:.2f}%")
    col2.metric("📋 Test Samples", f"{len(y_test):,}")
    col3.metric("✅ Correct Predictions",
                f"{(y_test == y_pred).sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Dissatisfied','Satisfied'],
                    yticklabels=['Dissatisfied','Satisfied'], ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Importance")
        feat = pd.Series(model.feature_importances_,
                         index=X_train.columns).sort_values()
        fig, ax = plt.subplots(figsize=(6, 8))
        feat.plot(kind='barh', color='steelblue', ax=ax)
        st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
                 target_names=['Dissatisfied','Satisfied'],
                 output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


# ════════════════════════════════════════════════════════════
# PAGE 5 – INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "💡 Insights & Recommendations":
    st.title("💡 Insights & Recommendations")
    st.markdown("---")

    st.subheader("🔑 Key Findings")
    findings = {
        "🥇 Online Boarding"      : "Strongest predictor of satisfaction",
        "🎬 Inflight Entertainment": "Critical for long-haul passengers",
        "💺 Seat Comfort"         : "Major driver for Economy class",
        "📶 Inflight Wifi"        : "High impact for Business travelers",
        "⏰ Departure Delays"     : "Longer delays = higher dissatisfaction",
        "👑 Business Class"       : "Most satisfied passenger segment",
    }
    for k, v in findings.items():
        st.info(f"**{k}** → {v}")

    st.markdown("---")
    st.subheader("🛠️ Recommendations")

    recs = [
        ("Online Boarding",
         "Streamline app check-in; add real-time gate notifications."),
        ("Inflight Entertainment",
         "Expand content library; ensure all screens work properly."),
        ("Seat Comfort",
         "Upgrade Economy seats; increase legroom on long-haul routes."),
        ("Inflight Wifi",
         "Provide free or affordable high-speed Wi-Fi especially in Business."),
        ("On-board Service",
         "Train crew for personalised and proactive passenger service."),
        ("Departure Delays",
         "Improve ground operations; notify passengers early via SMS/app."),
    ]

    for i, (area, tip) in enumerate(recs, 1):
        with st.expander(f"{i}. {area}"):
            st.write(f"✅ {tip}")

    st.markdown("---")
    st.subheader("📊 Satisfaction Rate by Segment")
    col1, col2 = st.columns(2)

    with col1:
        sat_class = df.groupby('Class')['satisfaction'].mean() * 100
        fig, ax   = plt.subplots()
        sat_class.plot(kind='bar', color='steelblue',
                       edgecolor='white', ax=ax)
        ax.set_title('Satisfaction % by Class')
        ax.set_ylabel('Satisfaction %')
        ax.set_xticklabels(['Business','Eco','Eco Plus'], rotation=0)
        st.pyplot(fig)

    with col2:
        sat_travel = df.groupby('Type of Travel')['satisfaction'].mean() * 100
        fig, ax    = plt.subplots()
        sat_travel.plot(kind='bar', color='teal',
                        edgecolor='white', ax=ax)
        ax.set_title('Satisfaction % by Travel Type')
        ax.set_ylabel('Satisfaction %')
        ax.set_xticklabels(['Business','Personal'], rotation=0)
        st.pyplot(fig)