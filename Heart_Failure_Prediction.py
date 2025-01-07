import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
# for logistic regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


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
        table, .dataframe {{
            background-color: black !important;
            color: white !important;
            border: 1px solid white !important;
        }}
        [data-testid="stSliderTickBarMin"] {{
            color: white !important;
        }}
         [data-testid="stSliderTickBarMax"] {{
            color: white !important;
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
    st.markdown(
        f"""<div style='background-color:black; color:white; padding:00px;'>
        {data.describe().to_html(classes='dataframe', index=True)}
        </div>""",
        unsafe_allow_html=True
    )

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
    
    if st.checkbox("Show Filtered Dataset"):
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
    fig.update_layout(
        title={"text": "Proportion of Death Events", "x": 0.46, "xanchor": "center", "font" :{"color":"white"}},
        legend = dict(font=dict(color="white")),
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ))
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
    fig.update_layout(
        title={"text": "Age Distribution with Death Event", "x": 0.5, "xanchor": "center", "font": {"color": "white"}},
        legend = dict(font=dict(color = "white")),
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender-based Death Proportions")
    gender_death = filtered_data.groupby("gender")["DEATH_EVENT"].mean() * 100
    fig = px.bar(
        gender_death,
        x=gender_death.index,
        y=gender_death.values,
        labels={ "x": "Gender", "y": "Death Proportion (%)"},
        title="Proportion of Death Events by Gender",
        color=gender_death.index,
        color_discrete_map={"Female": "pink", "Male": "green"}
    )
    fig.update_layout(
        title={"text": "Proportion of Death Events by Gender", "x": 0.5, "xanchor": "center", "font": {"color": "white"}},
        legend=dict(font=dict(color="white")),
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
        xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"))
    )
    st.plotly_chart(fig)

    st.subheader("High-risk Features")
    fig = px.box(
        filtered_data,
        x="DEATH_EVENT",
        y="ejection_fraction",
        color="DEATH_EVENT",
        labels={"DEATH_EVENT": "Death Event", "ejection_fraction":"Ejection Fraction"},
        title="Ejection Fraction by Death Event"
    )
    fig.update_layout(
        title={"text": "Ejection Fraction by Death Event", "x": 0.5, "xanchor": "center", "font": {"color": "white"}},
        legend = dict(font=dict(color = "white")),
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ))
    st.plotly_chart(fig)

    fig = px.violin(
        filtered_data,
        x="DEATH_EVENT",
        y="serum_creatinine",
        color="DEATH_EVENT",
        labels={"DEATH_EVENT": "Death Event", "serum_creatinine":"Serum Creatinine"},
        title="Serum Creatinine by Death Event"
    )
    fig.update_layout(
        title={"text": "Serum Creatinine by Death Event", "x": 0.5, "xanchor": "center", "font": {"color": "white"}},
        legend = dict(font=dict(color = "white")),
       paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ))
    st.plotly_chart(fig)

# Correlation Analysis page
elif selected_page == "Correlation Analysis":
    st.title("🔗 Correlation Analysis")

    st.subheader("Correlation Heatmap")
    st.write("This heatmap shows the correlation coefficients between numerical features, helping identify relationships.")

    def set_custom_style():
      
        st.markdown(
        """
        <style>
        /* Style the selectbox container background */
        div[data-baseweb="select"] > div {
            background-color: black !important;
            color: white !important;
        }

        /* Style the dropdown list when the selectbox is clicked */
        ul[role="listbox"] {
            background-color: black !important;
            color: white !important;
        }

        /* Style individual options in the dropdown */
        li[role="option"] {
            background-color: black !important;
            color: white !important;
        }

        /* Highlight the hovered option */
        li[role="option"]:hover {
            background-color: #333333 !important;
            color: white !important;
        }

        /* Style the dataframe container and scrollable block */
        div[data-testid="stDataFrameContainer"] {
            background-color: black !important;
            color: white !important;
        }

        /* Style the entire table content */
        .stDataFrame table {
            background-color: black !important;
            color: white !important;
        }

        /* Style table headers */
        .stDataFrame table thead th {
            background-color: #333333 !important;
            color: white !important;
            border-bottom: 1px solid white !important;
        }

        /* Style table rows */
        .stDataFrame table tbody tr {
            background-color: black !important;
            color: white !important;
        }

        /* Style hovered rows */
        .stDataFrame table tbody tr:hover {
            background-color: #444444 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Apply the custom style
    set_custom_style()

    # Filter out non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate the correlation matrix
    corr_matrix = numeric_data.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Correlation Matrix of Features")
    fig.update_layout(
        title={"text": "Correlation Matrix of Features", "x": 0.5, "xanchor": "center", "font": {"color": "white"}},
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
         xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),  # X-axis labels color
        yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"))   # Y-axis labels color
    )
    st.plotly_chart(fig)

    st.subheader("Top Correlated Features with DEATH_EVENT")
    death_corr = corr_matrix["DEATH_EVENT"].sort_values(ascending=False)
    death_corr_df = death_corr.to_frame()
    st.dataframe(death_corr, width=1230, height=400)

    st.subheader("Pairwise Feature Correlations")
    feature_x = st.selectbox("Select Feature X:", options=numeric_data.columns, index=0)
    feature_y = st.selectbox("Select Feature Y:", options=numeric_data.columns, index=1)

    fig = px.scatter(data, x=feature_x, y=feature_y, color="DEATH_EVENT",
                     title=f"Scatterplot of {feature_x} vs {feature_y}")
    fig.update_layout(
        title={"text": f"Scatterplot of {feature_x} vs {feature_y}", "x": 0.5, "xanchor": "center", "font": {"color": "white"}},
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font_color="white",
        xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),  # X-axis labels color
        yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"))  # Y-axis labels color

    )
    st.plotly_chart(fig)

    st.subheader("Feature Interaction Insights")
    st.write("This section highlights the most influential features and their interactions with DEATH_EVENT.")
    st.markdown("- **Ejection Fraction**: Negatively correlated with DEATH_EVENT. Patients with lower ejection fraction percentages are more likely to experience adverse outcomes. This highlights the importance of monitoring heart pump efficiency.")
    st.markdown("- **Serum Creatinine**: Positively correlated with DEATH_EVENT. Elevated serum creatinine levels indicate potential kidney dysfunction, which is associated with higher mortality risk.")
    st.markdown("- **Age**: Positively correlated with DEATH_EVENT. Older patients are more prone to severe outcomes, underlining the role of age in predicting health deterioration.")
    st.markdown("- **Serum Sodium**: Negatively correlated with DEATH_EVENT. Lower sodium levels in the blood are linked to higher mortality, emphasizing the significance of electrolyte balance.")
    st.markdown("- **Time**: Patients with shorter follow-up times are more likely to have experienced DEATH_EVENT, suggesting the need for early and continuous medical intervention.")

    st.write("Understanding these feature interactions provides actionable insights for clinicians to prioritize critical metrics and improve patient care.")

# ---------------------------------------------------------------------------------------------------------------------------------------
elif selected_page == "Survival Analysis":
    st.title("🕒 Survival Analysis")

    st.header("Introduction")
    st.write("Survival analysis is a statistical method used to study the time until an event of interest occurs. In healthcare, it is particularly valuable for analyzing outcomes like patient mortality, as it considers both the timing of the event and the possibility of incomplete observations (e.g., patients who were lost to follow-up).")
    st.write("In this app, we focus on predicting the time to DEATH_EVENT (1 = death, 0 = survival) for patients with heart failure. The dataset includes a time-related variable, time, which represents the number of days each patient was observed. Using survival analysis, we can explore patterns in survival probabilities and identify factors that influence patient outcomes over time. This insight is crucial for optimizing treatment strategies and improving care for at-risk patients.")

    # Define Kaplan-Meier model
    kmf = KaplanMeierFitter()
    st.subheader("Kaplan-Meier Survival Curve")
    st.write("The Kaplan-Meier Suvival Curve provides an estimate of survival probablities over time."
        "It is a crucial tool for understanding how long patients survive after diagnosis or treatment.")

    
    # Median survival time and survival curve
    survival_col = "time"  # Column representing time
    event_col = "DEATH_EVENT"  # Column representing the event (1 = death, 0 = survival)

    # Fit the Kaplan-Meier model for the entire dataset
    kmf.fit(data[survival_col], event_observed=data[event_col])
    fig = go.Figure()
    
    # Add the survival curve
    fig.add_trace(
        go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["KM_estimate"],
            mode="lines",
            name="Overall Survival",
            line=dict(color="red"),
        )
    )
    
    # Annotate the median survival time
    median_survival_time = kmf.median_survival_time_
    fig.add_trace(
        go.Scatter(
            x=[median_survival_time],
            y=[0.5],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=f"Median: {median_survival_time} days",
            textposition="bottom right",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": "Kaplan-Meier Survival Curve",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "white"}  # Set title font color to white
        },
        xaxis_title="Time (days)",
        yaxis_title="Survival Probability",
        paper_bgcolor="rgba(0, 0, 0, 0.3)",
        plot_bgcolor="rgba(0, 0, 0, 0.3)",
        font=dict(color="white"),
        xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white")),
        legend=dict(font=dict(color="white")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grouped Survival Analysis (e.g., by Gender)
    st.subheader("Survival by Gender")
    gender_groups = data["gender"].unique()
    fig_grouped = go.Figure()
    
    for gender in gender_groups:
        group_data = data[data["gender"] == gender]
        kmf.fit(group_data[survival_col], event_observed=group_data[event_col])
        fig_grouped.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_["KM_estimate"],
                mode="lines",
                name=f"Gender: {gender}",
                line=dict(width=2),
            )
        )

        fig_grouped.update_layout(
        title={
            "text": "Kaplan-Meier Survival Curve by Gender",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "white"}  # Ensure title font color is white
        },
        xaxis={
            "title": {"text": "Time (days)", "font": {"color": "white"}},  # X-axis title
            "tickfont": {"color": "white"},  # X-axis tick labels
            "showgrid": True,  # Show grid lines
            "gridcolor": "gray"  # Optional: Set gridline color
        },
        yaxis={
            "title": {"text": "Survival Probability", "font": {"color": "white"}},  # Y-axis title
            "tickfont": {"color": "white"},  # Y-axis tick labels
            "showgrid": True,  # Show grid lines
            "gridcolor": "gray"  # Optional: Set gridline color
        },
        paper_bgcolor="rgba(0, 0, 0, 0.3)",  # Background color
        plot_bgcolor="rgba(0, 0, 0, 0.3)",  # Plot area background color
        font=dict(color="white"),  # Overall font color
        legend=dict(font=dict(color="white")),  # Legend font color
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    # Summary Statistics
    st.subheader("Key Insights")
    st.markdown(f"- **Median Survival Time**: {median_survival_time} days")
    st.write(
        "The survival curve above shows the probability of survival over time for the entire dataset, "
        "as well as stratified by gender. This helps identify high-risk groups."
    )

    st.subheader("⏸️ Event Censoring")
    st.markdown("In survival analysis, not all patients experience the event of interest (e.g., death) during the study period."
            "These patients are considered **censored**, meaning their exact survival time is unknown. Censoring occurs when a patient:")
    st.markdown("- Survives beyond the observation period.")   
    st.markdown("- Is lost to follow-up before the event occurs.")    
    st.markdown("Censoring is a common phenomenon in medical datasets and is critical to account for when analyzing survival probabilities.")
    st.write("**How Censoring is Handled**")
    st.write("The Kaplan-Meier Survival Curve accounts for censored data by estimating the survival probability only for the time intervals where complete information is available."
             "Censored individuals are included in the calculations up until their last recorded time, ensuring that the survival curve is as accurate as possible.")
    st.write("By properly accounting for censored data, the Kaplan-Meier method provides a robust estimate of survival probabilities, even when the observation period is incomplete for some patients.")     
    
# -----------------------------------------------------------------------------------------------------------------------------------
elif selected_page == "Model Prediction":
    # ML Model ==> LOGISTIC REGRESSION
   # Encode categorical variables 
    X = data.drop(columns=["DEATH_EVENT"])  # Exclude the target variable
    y = data["DEATH_EVENT"]                # Target variable

    # Convert categorical columns to numeric (e.g., "gender")
    X = pd.get_dummies(X, columns=["gender"], drop_first=True)  # Ensuring only gender_Male is used as a column

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Streamlit display
    st.subheader("Logistic Regression Results")
    st.write(f"Model Accuracy: {accuracy:.2f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Survived", "Death"], yticklabels=["Survived", "Death"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")

    # Display the classification report using Streamlit's built-in code function
    st.code(class_report, language="plaintext")

    # Make predictions on new data
    st.subheader("Make Predictions")
    user_input = {}

    # Update input columns to match the encoded feature names (gender_Male instead of gender)
    for column in X.columns:
        if column == "gender_Male":
            # Directly assign the value for gender_Male based on a predefined assumption (no user input required)
            user_input[column] = 1  # Assuming Male as the input, modify as per your requirement
        else:
            user_input[column] = st.number_input(f"Enter {column}:", value=float(data[column].mean()))

    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    st.write(f"Predicted Outcome: {'Death' if prediction == 1 else 'Survived'}")
    st.write(f"Prediction Probability: Death - {prediction_proba[1]:.2f}, Survived - {prediction_proba[0]:.2f}")
