# HandPlay — Rock, Paper, Scissors

A Streamlit web app that recognises rock, paper, and scissors hand gestures from a camera snapshot or uploaded image.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload this folder's contents.
2. At [share.streamlit.io](https://share.streamlit.io), select **Create app** and connect the repository.
3. Select `app.py` as the main file and click **Deploy**.

`runtime.txt` pins Python 3.11 for MediaPipe compatibility. The training dataset is not required for deployment.

## How it works

MediaPipe detects 21 hand landmarks. Their coordinates are centered on the wrist and scaled to hand size, then a logistic-regression model predicts the gesture.
