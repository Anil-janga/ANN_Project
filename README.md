✍️ Handwritten Digit Recognition using Pygame & Deep Learning

This project allows users to draw digits (0–9) on a screen using a mouse, and a trained deep learning model predicts the digit in real time.

📌 Features
Draw digits using mouse
Real-time digit prediction
Uses trained Keras model (.keras)
Displays predicted digit on screen
Option to save drawn images
🛠️ Technologies Used
Python
Pygame (for drawing interface)
NumPy
OpenCV
Keras / TensorFlow
📂 Project Structure
digit-recognition/
│
├── main.py              # Main application file
├── bestmodel.keras      # Pre-trained model
└── README.md            # Documentation
⚙️ How It Works
User draws a digit using the mouse
Coordinates of drawing are recorded
Drawing area is cropped automatically
Image is resized to 28x28 pixels
Image is normalized and passed to model
Model predicts the digit (0–9)
Result is displayed on the screen

🎮 Controls
🖱️ Hold Mouse → Draw digit
🖱️ Release Mouse → Predict digit
⌨️ Press N → Clear screen

