import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Configure the Streamlit page
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="heart.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the dataset
@st.cache_data
def sample_data():
    df = pd.read_csv("Dataset/heart_failure_clinical_records_dataset.csv")
    return df
# Apply custom CSS styles
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
        .stApp {{
            background-image: url("https://www.cardio.com/wp-content/uploads/2024/03/CIS-1-1-2-March-1024x576.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Sidebar customization */
        .sidebar .sidebar-content { 
            background-color: #ffffff; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        }
        /* Title customization */
        h1 { 
            color: rgb(8, 9, 9) !important; /* Black color for title */
            font-size: 36px; 
            font-family: 'Source Sans Pro', sans-serif !important;
            font-weight: bold; 
            margin-bottom: 10px; 
        }
        /* Subheading customization */
        h2, h3 { 
            font-family: 'Source Sans Pro', sans-serif !important;
            color: rgb(10, 9, 9); /* Black for subheadings */
            font-weight: bold; 
            font-size: 15px;  /* Font size reduced for subheaders */
        }
        /* Header (overview) customization */
        h2 { 
            font-family: 'Source Sans Pro', sans-serif !important;
            color: rgb(10, 9, 9) !important; /* Black color for headers */
            font-size: 28px; 
            font-weight: bold; 
            margin-top: 20px;
        }
        h3 {
            font-family: 'Source Sans Pro', sans-serif !important;
            color: rgb(10, 9, 9) !important; /* Black color for sub-headers */ 
            font-weight: bold; 
            margin-top: 20px;
        }
        /* Paragraph text styling */
        p { 
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 16px; 
            color: rgb(10, 9, 9); /* Black for paragraph text */
        }
        /* Footer customization */
        .footer { 
            font-family: 'Source Sans Pro', sans-serif !important;
            margin-top: 20px; 
            text-align: center; 
            font-size: 14px; 
            color: rgb(10, 9, 9); /* Black for footer text */
        }
        /* Card styling */
        .card { 
            font-family: 'Source Sans Pro', sans-serif !important;
            background-color: #ffffff; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
            margin-bottom: 20px; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

df = sample_data()


# Page title
st.title("🫀 Heart Failure Prediction")

# sidebar
st.sidebar.header("Explore Analysis")
# sidebar options
options = [
    "🕵🏼‍♂️ Overview",
    "🩸Anaemic Factor",
    "Impact of Speed",
    "Alcohol Influence",
    "First-Time Offender",
    "High-Risk States"
]
sb_option = st.sidebar.radio("Choose an analysis:", options)
# ------------------------------------------------------------------------------------------------------------------------------------------------
# page 1 ==> Overview
if sb_option == "🕵🏼‍♂️ Overview":
    
    st.write("Predicting heart failure risks using data-driven insights for better healthcare outcomes.")

    # Overview section
    st.subheader("Overview")
    st.write(
    "This application leverages advanced data analysis to predict the risk of heart failure. "
    "By analyzing clinical data, users can gain valuable insights to support early diagnosis "
    "and improve patient outcomes."
    )
    st.subheader("📊 Summary")
    avg_deaths = round((df['DEATH_EVENT'].mean()) * 100, 2)
    avg_males_death = round((df[df["sex"] == 1]["DEATH_EVENT"].mean()) * 100, 2)
    avg_female_death = round((df[df["sex"] == 0]["DEATH_EVENT"].mean()) * 100, 2)
    st.metric(label="⚰️ Average Deaths", value=f"{avg_deaths} %")
    st.metric(label="♂️ Average Male Deaths", value=f"{avg_males_death} %")
    st.metric(label="♀️ Avergae Female Deaths", value=f"{avg_female_death} %")
    # option for the user to toggle between full datset, and just the head rows
    option_toggle = st.radio("📁 Dataset Preview: ", ["Preview (Top 5 Rows)", "Full Dataset"])
    if option_toggle == "Preview (Top 5 Rows)":
        st.dataframe(df.head())
    else:
        st.dataframe(df)

# ------------------------------------------------------------------------------------------------------------------------------------------------

# page 2 ==> Anaemia
if sb_option == "🩸Anaemic Factor":
    st.header("🩸 Anaemia")
    st.write("A blood disorder that occurs when your body doesn't have enough healthy red blood cells or hemoglobin to carry oxygen throughout your body.")
    st.subheader("🚨  Symptoms")
    st.write("Common symptoms include fatigue, weakness, pale skin, shortness of breath, and dizziness. It can result from various causes, including nutritional deficiencies (such as iron, vitamin B12, or folate), chronic diseases, genetic disorders like sickle cell anemia, or blood loss due to injury, menstruation, or internal bleeding. Anemia affects people of all ages and genders but is particularly prevalent in women of childbearing age, pregnant women, and children. Diagnosis typically involves blood tests to assess hemoglobin levels and identify underlying causes. Treatment depends on the type and cause of anemia and may include dietary changes, supplements, medications, or addressing the root condition. Early detection and management are crucial to prevent complications and improve quality of life.")
    st.image("anaemia.jpg", caption="Symptoms of Anaemia", width=700)
    # Key stats
    st.subheader("📶 Key Statistics")

    # Ensure the replacement operation is applied correctly
    df["gender"] = df["sex"].map({0: "Female", 1: "Male"})
    # Create a selectbox to choose between male and female death rate
    selected_gender = st.selectbox("Select Option", ["Combined", "Gender Wise"])
    
# ------------------------------------------------------------------------------------------------------------------------------------------------
    # Gender Wise Stats
    if selected_gender == "Gender Wise":
    # Calculate and display the average death percentage for male anaemic patients
        male_avg_an_death = round(df[(df["gender"] == "Male") & (df["anaemia"] == 1)]["DEATH_EVENT"].mean() * 100, 2)
        st.metric(label="Average Male Deaths Due to Anaemia (%)", value=f"{male_avg_an_death}%")
        female_avg_an_death = round(df[(df["gender"] == "Female") & (df["anaemia"] == 1)]["DEATH_EVENT"].mean() * 100, 2)
        st.metric(label="Average Female Deaths Due to Anaemia (%)", value=female_avg_an_death)
    

# ------------------------------------------------------------------------------------------------------------------------------------------------:
                                    # Visualize male and female death rates separately on thr bar chart
    
        fig = go.Figure()
        # ADD male Data
        fig.add_trace(go.Bar(
            x=["Male"], y=[male_avg_an_death], name="Male", marker_color = "blue", text=[f"{male_avg_an_death:.2f} %"], 
            textposition="auto"))
        # ADD Female Data
        fig.add_trace(go.Bar(
            x=["Female"], y=[female_avg_an_death], name="Female", marker_color="pink", text=[f"{female_avg_an_death:.2f} %"], 
            textposition="auto"))
        # update Layout
        fig.update_layout(
            title={"text": "Male and Female Death Rates", "x": 0.5,"xanchor": "center"},
            xaxis_title = "Gender", 
            yaxis_title = "Death Rate (%)", 
            yaxis= dict(range=[0, 100]), 
            barmode = "group")
        # display the plot
        st.plotly_chart(fig)


# ------------------------------------------------------------------------------------------------------------------------------------------------

    # if both combined
    else:
        # anaemic deaths
        anaemic_deaths = round((df[df["anaemia"] == 1]["DEATH_EVENT"].mean()) * 100, 2)

        st.metric(label="Average Deaths Due to Anaemia", value=f"{anaemic_deaths} %")

        # Filter data for anaemic patients
        anaemia_death_data = (df[df['anaemia'] == 1])

        # Count the outcomes for DEATH_EVENT among anaemic patients and calculate percentages
        death_counts = anaemia_death_data['DEATH_EVENT'].value_counts(normalize=True).reset_index()
        death_counts.columns = ['Death Event', 'Proportion']  # Rename columns
        death_counts['Percentage'] = death_counts['Proportion'] * 100  # Calculate percentages

        # Replace numeric values in 'Death Event' with labels
        death_counts['Death Event'] = death_counts['Death Event'].replace({0: 'No', 1: 'Yes'})

        # Create a pie chart
        fig = px.pie(death_counts, 
                values='Percentage', 
                names='Death Event', 
                title='Percentage of Deaths Among Anaemic Patients',
                color='Death Event', 
                color_discrete_map={'Yes': 'red', 'No': 'green'})
        
        # Customize the layout for the title and add text at the bottom
        fig.update_layout(
        title={
            'text': 'Average number of Deaths Among Anaemic Patients (Percentage)',
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        annotations=[
            dict(
                text="On an Average, 35.7 % heart failures resulting to death are due to anaemia",
                showarrow=False,
                x=0.5,  # Center text at the bottom
                y=-0.1,
                xref="paper",
                yref="paper",
                font=dict(size=12),
                align="center"
            )
        ]
    )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
    

    
   











    