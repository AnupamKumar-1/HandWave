# 🤚 HandWave – Real-Time Hand Gesture Recognition

**HandWave** is a computer vision and machine learning project that allows you to control actions using hand gestures captured through a webcam. It’s lightweight, simple, and containerized for easy deployment.

<table>
<tr>
<td width="50%">

### 🚀 Features

- 🎥 Real-time gesture recognition  
- 🧠 Custom-trained model  
- 📦 Dockerized deployment  
- 🗂️ Data collection & testing scripts  

</td>
<td width="50%">

### 🧰 Tech Stack

- Python  
- OpenCV  
- Mediapipe  
- scikit-learn  
- Docker  

</td>
</tr>
</table>



## Control Flow 

```mermaid
sequenceDiagram
  participant U as User
  participant C as Webcam
  participant CV as Computer Vision
  participant ML as Model
  participant F as Frontend (Web)
  participant B as Backend (Flask)
  
  U->>C: Perform hand gesture
  C-->>CV: Capture frame
  CV->>CV: Detect and extract hand landmarks
  CV->>ML: Send features for prediction
  ML-->>CV: Return gesture label
  CV->>B: Send label to backend
  B->>F: Send prediction response
  F-->>U: Display gesture result
```

---


## 🏗️ Architecture Diagram

```mermaid
graph LR
  subgraph Client
    A[User] --> B[Webcam]
    B --> C[Hand Detection - MediaPipe or OpenCV]
    C --> D[Feature Extraction]
    D --> E[Frontend - JavaScript]
  end

  subgraph Server
    E --> F[Backend - Flask]
    F --> G[Load Model - model.p]
    G --> H[Predict Gesture]
    H --> I[Return Prediction]
  end

  I --> E
  E --> A

  subgraph Deployment
    F --> J[Docker Container]
    J --> K[Cloud or Local Host]
  end

```

---


## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/HandWave.git
cd HandWave
```
### Install Dependencies (Without Docker)
Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install required packages

```bash
pip install -r requirements.txt
cd webapp
pip install -r requirements.txt
```

### 3. Collect Hand Gesture Data

```bash
python collecting_data.py
```

### 4. Process Data

```bash
processing_data.py
```
### 5. Train the Model

```bash
python model_train.py
```
### 6. Run the Web App
```bash
cd webapp

python app.py
```

Then open your browser and visit:

```bash
http://localhost:5000
```

## 🐳 Docker Setup (Optional)

### Build Docker Image

```bash
docker build -t handwave .
```

### Run Docker Container

```bash
docker run -p 5000:5000 handwave
```
```bash
Access the app at: http://localhost:5000
```
## 📄 License

This project is licensed under the MIT License.

## 🙌 Contributions

Feel free to open issues or submit pull requests to improve this project!

