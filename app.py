import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

# Load the model
model = load_model("my_model.h5")

# Define feature names
FEATURES = ['Port', 'Protocol', 'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts']

# Set up SQLite database
def init_db():
    conn = sqlite3.connect("flagged_predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS flagged_predictions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 port INTEGER,
                 protocol INTEGER,
                 flow_duration REAL,
                 tot_fwd_pkts INTEGER,
                 tot_bwd_pkts INTEGER,
                 label TEXT,
                 confidence REAL,
                 timestamp TEXT
                 )''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Function to create a bar chart for confidence
def create_confidence_bar(confidence):
    fig, ax = plt.subplots(figsize=(4, 3))  # Smaller size for Gradio
    ax.bar(["Confidence"], [confidence], color="blue" if confidence > 0.5 else "green", width=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Prediction Confidence")
    plt.tight_layout()
    return fig

# Prediction function with error handling
def predict_traffic(port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts):
    try:
        input_data = np.array([[port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts]])
        reshaped_input = np.zeros((1, 10, 77))
        reshaped_input[0, 0, :5] = input_data[0]
        
        prediction = model.predict(reshaped_input)
        label = "Malicious" if prediction[0][0] > 0.5 else "Benign"
        confidence = float(prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0])  # Convert numpy.float64 to float
        
        return (
            f"Prediction: {label}",
            create_confidence_bar(confidence),
            gr.update(visible=True),
            port,
            protocol,
            flow_duration,
            tot_fwd_pkts,
            tot_bwd_pkts,
            label,
            confidence  # Raw confidence value for flagging
        )
    except Exception as e:
        return (
            "Error in prediction",
            None,
            gr.update(visible=False),
            None,
            None,
            None,
            None,
            None,
            None,
            0  # Default confidence value in case of error
        )

# Function to save flagged data to database
def flag_prediction(port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts, label, confidence):
    if port is None:  # Skip if prediction failed
        return "Cannot flag: Prediction failed"
    conn = sqlite3.connect("flagged_predictions.db")
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO flagged_predictions (port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts, label, confidence, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts, label, confidence, timestamp))
    conn.commit()
    conn.close()
    return "Flagged successfully!"

# Custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
).set(
    body_background_fill="#f0f2f6",
    button_primary_background_fill="#1e40af",
    button_primary_text_color="#ffffff",
    block_title_text_weight="600",
    block_border_width="2px",
)

# Create the app with Blocks
with gr.Blocks(theme=theme, title="CyberShield: AI-Powered Threat Defense") as app:
    # Header/Navbar
    with gr.Row(elem_id="navbar", variant="compact"):
        gr.Markdown(
            """
            # CyberShield
            *AI-Powered Threat Defense*
            """,
            elem_id="header-title"
        )
        with gr.Row():
            home_btn = gr.Button("Home", size="sm", variant="primary")
            predict_btn = gr.Button("Prediction", size="sm", variant="primary")
            dataset_btn = gr.Button("Dataset", size="sm", variant="primary")
            about_btn = gr.Button("About", size="sm", variant="primary")

    # Main content
    with gr.Column(variant="panel"):
        # Home Page
        with gr.Group(visible=True) as home_page:
            gr.Markdown("## Welcome to CyberShield", elem_classes="section-title")
            gr.Image("dashboard.jpg", show_label=False, container=False)
            gr.Markdown("""
            An advanced AI-powered solution for network threat detection.
            
            ### Key Features:
            - **Real-time Analysis**: Monitor traffic instantly
            - **High Accuracy**: Advanced neural network predictions
            - **Intuitive UI**: Easy-to-use interface
            
            Navigate using the top bar to explore more.
            """)
            try_predict_btn = gr.Button("Try Prediction Now", variant="primary")

        # Prediction Page
        with gr.Group(visible=False) as predict_page:
            gr.Markdown("## Traffic Prediction", elem_classes="section-title")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Features")
                    port = gr.Number(label="Port", info="Network port number")
                    protocol = gr.Number(label="Protocol", info="Communication protocol")
                    flow_duration = gr.Number(label="Flow Duration", info="Connection duration")
                    tot_fwd_pkts = gr.Number(label="Total Forward Packets", info="Packets sent")
                    tot_bwd_pkts = gr.Number(label="Total Backward Packets", info="Packets received")
                    predict_submit = gr.Button("Analyze Traffic", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### Results")
                    pred_output = gr.Textbox(label="Prediction", interactive=False)
                    conf_plot = gr.Plot(label="Confidence")
                    flag_btn = gr.Button("Flag", variant="secondary")
                    flag_status = gr.Textbox(label="Flag Status", interactive=False)
                    status = gr.State()
                    # Hidden state to store raw confidence for flagging
                    confidence_state = gr.State()
                    with gr.Row(visible=False) as result_details:
                        gr.Markdown("*Analysis completed successfully*")
            
            # Prediction event
            predict_submit.click(
                fn=predict_traffic,
                inputs=[port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts],
                outputs=[
                    pred_output,           # Prediction text
                    conf_plot,            # Bar chart
                    result_details,       # Success message
                    port,                 # Persist input
                    protocol,             # Persist input
                    flow_duration,        # Persist input
                    tot_fwd_pkts,         # Persist input
                    tot_bwd_pkts,         # Persist input
                    pred_output,          # Persist label
                    confidence_state      # Store raw confidence
                ]
            )
            
            # Flag event
            flag_btn.click(
                fn=flag_prediction,
                inputs=[port, protocol, flow_duration, tot_fwd_pkts, tot_bwd_pkts, pred_output, confidence_state],
                outputs=[flag_status]
            )

        # Dataset Page
        with gr.Group(visible=False) as dataset_page:
            gr.Markdown("## Dataset Details", elem_classes="section-title")
            gr.Markdown("""
            ### Features Used:
            | Feature | Description |
            |---------|-------------|
            | Port | Network port number |
            | Protocol | Communication protocol |
            | Flow Duration | Duration of the connection |
            | Total Forward Packets | Number of packets sent |
            | Total Backward Packets | Number of packets received |
            
            ### Model Specifications:
            - **Architecture**: Deep Neural Network
            - **Input Shape**: (10, 77)
            - **Training Data**: This dataset was originally created by the University of New Brunswick for analyzing DDoS data. You can find the full dataset here. This dataset was sourced fully from 2018, and will not be updated in the future, however, new versions of the dataset will be available at the link above. The dataset itself was based on logs of the university's servers, which found various DoS attacks throughout the publicly available period. When writing machine learning notebooks for this data, note that the Label column is arguably the most important portion of data, as it determines if the packets sent are malicious or not. Reference the below Column Structures heading for more information about this and more columns.
            - **Accuracy**: 0.7989
            - **Loss**: 0.4330
            """)
            gr.Button("Download Sample Data", link="https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv")

        # About Page
        with gr.Group(visible=False) as about_page:
            gr.Markdown("## About CyberShield", elem_classes="section-title")
            gr.Markdown("""
            ### Our Mission:
            Protecting networks through cutting-edge AI technology.
            
            ### Technology Stack:
            - **TensorFlow**: Deep learning framework
            - **Gradio**: Interactive UI deployment
            - **Python**: Core development language
            
            ### Team:
            Developed by [Nextgen Solutions]
            
            Contact us: [your.email@example.com](mailto:your.email@example.com)
            """)
            gr.HTML("""
            <div style='text-align: center; margin-top: 20px;'>
                <a href='https://github.com' target='_blank'>GitHub</a> | 
                <a href='https://twitter.com' target='_blank'>Twitter</a>
            </div>
            """)

    # Footer
    with gr.Row(variant="compact"):
        gr.Markdown(
            f"© {2025} CyberShield. All rights reserved. | Version 1.0.0",
            elem_classes="footer"
        )

    # Navigation logic
    def show_page(page):
        return {
            home_page: gr.Group(visible=page == "home"),
            predict_page: gr.Group(visible=page == "predict"),
            dataset_page: gr.Group(visible=page == "dataset"),
            about_page: gr.Group(visible=page == "about")
        }

    # Connect buttons to navigation
    home_btn.click(fn=lambda: show_page("home"), inputs=None, outputs=[home_page, predict_page, dataset_page, about_page])
    predict_btn.click(fn=lambda: show_page("predict"), inputs=None, outputs=[home_page, predict_page, dataset_page, about_page])
    dataset_btn.click(fn=lambda: show_page("dataset"), inputs=None, outputs=[home_page, predict_page, dataset_page, about_page])
    about_btn.click(fn=lambda: show_page("about"), inputs=None, outputs=[home_page, predict_page, dataset_page, about_page])
    try_predict_btn.click(fn=lambda: show_page("predict"), inputs=None, outputs=[home_page, predict_page, dataset_page, about_page])

# Launch the app
app.launch()
