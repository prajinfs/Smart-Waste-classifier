♻️ Smart Waste Classifier

An AI-powered waste classification application that predicts the category of waste from an uploaded image.

Features

- Upload waste images in JPG, JPEG, or PNG format.
- Classify waste into six categories:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
- Display the predicted category and confidence score.
- Show category-wise prediction probabilities.

Technologies Used

- Python
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow

Project Files

- "app.py" — Runs the web application.
- "train_model.py" — Trains the image classification model.
- "class_names.txt" — Contains the category names.
- "waste_classifier.keras" — Saved model.

How to Run Locally

1. Install Python 3.12.

2. Install the required packages:
   
   pip install streamlit tensorflow numpy pillow

3. Start the application:
   
   streamlit run app.py

4. Open the local URL displayed in your terminal.

Note

Predictions depend on the training data and model performance. Always verify the waste category before recycling.

Author

Prajin F.S
