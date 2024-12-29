import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Configure the Streamlit page
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="heart.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styles with background image
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
        .stApp {{
            background-image: url("https://d2jx2rerrg6sh3.cloudfront.net/images/news/ImageForNews_784134_17197968792946306.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgb(183, 6, 12) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] .css-1d391kg {{
            padding: 15px !important;
        }}
        h1, h2, h3, p, .css-1d391kg {{
            color: #FFFFFF !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            text-align: center !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: red !important;
        }}
        li {{
            color: #FFFFFF !important;
        }}
        .css-1fcdlhh, .css-1q8dd3e {{
            color: #FFFFFF !important;
        }}
        table {{
            background-color: black !important;
            color: white !important;
            border: 1px solid white !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Dataset/heart_failure_clinical_records_dataset.csv")

data = load_data()
data["gender"] = data["sex"].map({0: "Female", 1: "Male"})

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Overview", "Key Insights", "Correlation Analysis", "Survival Analysis", "Model Prediction"]
selected_page = st.sidebar.radio("Go to", pages)

# Overview page
if selected_page == "Overview":
    st.title("🫀 Heart Failure Prediction")
    st.write("This app analyzes clinical data to predict heart failure risks and offers data-driven insights for better healthcare decisions.")

    st.subheader("Dataset Overview")
    st.markdown("### Highlighted Features:")
    st.markdown("- **Age**: Represents the age of the patient.")
    st.markdown("- **Ejection Fraction**: Indicates the percentage of blood leaving the heart during each contraction.")
    st.markdown("- **Serum Creatinine**: A key marker for kidney function.")
    st.markdown("- **DEATH_EVENT**: Binary indicator of survival (1 = death, 0 = survival).")

    if st.checkbox("Show Dataset"):
        st.markdown(
            data.to_markdown(tablefmt="html"),
            unsafe_allow_html=True
        )

    st.subheader("Summary Statistics")
    st.write("This section provides descriptive statistics for all numerical features, highlighting central tendencies and variability:")
    st.write(data.describe())

    st.subheader("Interactive Data Exploration")
    st.write("Use the filters below to explore subsets of the data:")
    age_range = st.slider(
        "Select Age Range:", 
        int(data.age.min()), 
        int(data.age.max()), 
        (40, 60),
        format="%d",
        key="age_slider"
    )
    st.markdown(f"<p style='color: white;'>Range: {age_range[0]} to {age_range[1]}</p>", unsafe_allow_html=True)
    filtered_data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]
    st.write(f"Filtered data for age range {age_range}: ")
    st.markdown(
        filtered_data.to_markdown(tablefmt="html"),
        unsafe_allow_html=True
    )

# Key Insights page
elif selected_page == "Key Insights":
    st.title("📊 Key Insights")

    st.subheader("Interactive Filtering")
    st.write("Use the filters below to customize the insights:")

    gender_filter = st.multiselect("Select Gender:", options=data["gender"].unique(), default=data["gender"].unique())
    age_range = st.slider("Select Age Range:", int(data.age.min()), int(data.age.max()), (40, 60))

    filtered_data = data[(data["gender"].isin(gender_filter)) & (data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]
    st.write(f"Filtered data based on selected criteria:")
    st.markdown(filtered_data.to_markdown(tablefmt="html"), unsafe_allow_html=True)

    st.subheader("Death Event Proportions")
    death_proportion = filtered_data["DEATH_EVENT"].value_counts(normalize=True) * 100
    fig = px.pie(
        names=["Survived", "Death"],
        values=death_proportion,
        color=["Survived", "Death"],
        color_discrete_map={"Survived": "green", "Death": "red"},
        title="Proportion of Death Events"
    )
    fig.update_layout(title={"text": "Proportion of Death Events", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age Distribution")
    fig = px.histogram(
        filtered_data,
        x="age",
        nbins=20,
        color="DEATH_EVENT",
        labels={"DEATH_EVENT": "Death Event"},
        title="Age Distribution with Death Event"
    )
    fig.update_layout(title={"text": "Age Distribution with Death Event", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender-based Death Proportions")
    gender_death = filtered_data.groupby("gender")["DEATH_EVENT"].mean() * 100
    fig = px.bar(
        gender_death,
        x=gender_death.index,
        y=gender_death.values,
        labels={"y": "Death Proportion (%)", "x": "Gender"},
        title="Proportion of Death Events by Gender"
    )
    fig.update_layout(title={"text": "Proportion of Death Events by Gender", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

    st.subheader("High-risk Features")
    fig = px.box(
        filtered_data,
        x="DEATH_EVENT",
        y="ejection_fraction",
        color="DEATH_EVENT",
        labels={"DEATH_EVENT": "Death Event"},
        title="Ejection Fraction by Death Event"
    )
    fig.update_layout(title={"text": "Ejection Fraction by Death Event", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

    fig = px.violin(
        filtered_data,
        x="DEATH_EVENT",
        y="serum_creatinine",
        color="DEATH_EVENT",
        labels={"DEATH_EVENT": "Death Event"},
        title="Serum Creatinine by Death Event"
    )
    fig.update_layout(title={"text": "Serum Creatinine by Death Event", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

# Correlation Analysis page
elif selected_page == "Correlation Analysis":
    st.title("🔗 Correlation Analysis")

    st.subheader("Correlation Heatmap")
    corr_matrix = data.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Correlation Matrix of Features")
    fig.update_layout(title={"text": "Correlation Matrix of Features", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

    st.subheader("Top Correlated Features with Death Event")
    death_corr = corr_matrix["DEATH_EVENT"].sort_values(ascending=False)
    st.write(death_corr)

# Survival Analysis page
elif selected_page == "Survival Analysis":
    st.title("⏳ Survival Analysis")

    st.subheader("Survival by Ejection Fraction")
    fig = px.box(data, x="DEATH_EVENT", y="ejection_fraction", color="DEATH_EVENT",
                 labels={"DEATH_EVENT": "Death Event"},
                 title="Ejection Fraction and Survival")
    fig.update_layout(title={"text": "Ejection Fraction and Survival", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

    st.subheader("Survival by Serum Creatinine")
    fig = px.violin(data, x="DEATH_EVENT", y="serum_creatinine", color="DEATH_EVENT",
                    labels={"DEATH_EVENT": "Death Event"},
                    title="Serum Creatinine Levels and Survival")
    fig.update_layout(title={"text": "Serum Creatinine Levels and Survival", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

# Model Prediction page
elif selected_page == "Model Prediction":
    st.title("🤖 Heart Failure Prediction Model")

    st.subheader("Train a Model")
    target = "DEATH_EVENT"
    features = data.drop(columns=["DEATH_EVENT", "sex", "gender"])

    X_train, X_test, y_train, y_test = train_test_split(features, data[target], test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    st.write("### Classification Report")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    st.subheader("Feature Importance")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features.columns, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance", labels={"Importance": "Importance Score"})
    fig.update_layout(title={"text": "Feature Importance", "x": 0.5, "xanchor": "center"})
    st.plotly_chart(fig)

    st.write("### Make a Prediction")
    user_inputs = {col: st.number_input(f"Enter {col}", value=float(data[col].mean())) for col in features.columns}
    prediction = model.predict(np.array(list(user_inputs.values())).reshape(1, -1))[0]

    if prediction == 0:
        st.success("The model predicts no heart failure risk.")
    else:
        st.error("The model predicts a heart failure risk.")
