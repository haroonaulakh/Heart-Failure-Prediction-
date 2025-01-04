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
