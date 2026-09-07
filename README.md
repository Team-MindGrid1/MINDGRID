
# Mind Grid # MindGrid — Multimodal Assistive Communication System

## About Project
Mind Grid is an AI-powered system that translates Sign Language into Urdu text and then converts it to Speech. 
It helps bridge the communication gap for the deaf and mute and paralyze community in Pakistan.


## Team Members
- Sajjal Tasleem
- Urooj Saira
- Eshmaal Hashmi
- Esha Javed
- Khair-un-Nisa

## Features
- Sign Language Detection
- Urdu Text Translation 
- Text to Speech in Urdu


## 🚀 System Features
- **Ocular Tracking**: ParallelIZED Python pipelines utilizing MediaPipe Face Landmarker to process horizontal and vertical iris offsets.
- **Multimodal Coordination**: Blends eye movements (40%) and head turns (60%) for highly natural, accessible selections.
- **Dual-Model Hand Gesture Pipeline**: Leverages MediaPipe GestureRecognizer & HandLandmarker for fallback finger counting to achieve 10 discrete manual commands.
- **Urdu Text-to-Speech**: Instantly maps 16 separate gaze and gesture inputs to real-time Urdu speech output.

## 🛠️ How to Run the Software
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the live camera system demo:
   ```bash
   python live_demo.py --camera webcam
   ```
4. Run the offline simulated testing harness (uses keyboard arrow controls):
   ```bash
   python live_demo.py
   ```
