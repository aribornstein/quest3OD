# Real-Time Object & Body-Part Tracking in WebXR Using Lightweight Contrastive Models

## Overview

This project demonstrates **real-time object and body-part tracking** in the browser using WebXR, powered by a custom-trained, lightweight vision encoder (MobileNetV3-Small backbone) with contrastive learning. The pipeline is optimized for edge devices like the Meta Quest 3, focusing on speed, interactivity, and practical accuracy.

**Key features:**

* Efficient, custom object and keypoint detection/tracking with browser-friendly models.
* WebXR integration for AR/VR interaction and physics (via [Rapier](https://rapier.rs/)).
* Contrastive CLIP-style training to maximize model capability for low-resource deployment.
* Full open-source and browser-compatible toolchain (TensorFlow\.js / ONNX.js / MediaPipe).

---

## Project Goals

* **Real-time object/body-part detection and tracking** in WebXR.
* **Efficient on-device inference** (edge, no server required).
* **Demonstrate novel training:** Leverage contrastive CLIP-style objectives with MobileNetV3 for small, fast models.
* **Enable interaction with virtual objects** in WebXR using detected body/object locations.

---

## Pipeline Steps

### 1. Dataset Preparation

* Source: [Charades-Ego dataset](https://prior.allenai.org/projects/charades-ego)
* Selected a small subset of paired first-person and third-person videos (\~10–20 pairs for prototyping).
* Downloaded videos and associated annotation files.
* Extracted video frames and relevant annotation data.

### 2. Object and Body-Part Annotation Processing

* Ran pre-trained SOTA models (e.g., YOLOv8, Segment Anything, MediaPipe Hands/Pose) on video frames to generate segmentation masks and object bounding boxes.
* Saved processed annotations in a consistent format (COCO or custom JSON).

### 3. Multi-Object Tracking

* Applied [SORT](https://github.com/abewley/sort), DeepSORT, or ByteTrack algorithms to assign unique track IDs to detected objects and body parts across video frames.
* Generated per-frame object/track data for later training.

### 4. Vision Encoder Training

* Built a CLIP-style training pipeline using **MobileNetV3-Small** as the vision encoder and a lightweight text encoder.
* Used **contrastive loss** to align image (frame/object crop) and caption embeddings.
* Trained on the subset dataset, leveraging action and object labels/captions.
* Evaluated and exported the best-performing model.

### 5. Model Optimization for Edge Deployment

* Quantized and converted the trained model to **TensorFlow\.js** or **ONNX.js** format.
* Tested inference speed and accuracy in browser and on Quest 3.
* Fine-tuned model size and quantization for optimal real-time performance.

### 6. WebXR Integration and Real-Time Tracking

* Integrated the model into a WebXR scene using Three.js/A-Frame/Babylon.js.
* Ran inference on camera frames, updating object/body-part positions in real time.
* Linked detection results to virtual objects with [Rapier](https://rapier.rs/) for physics-based interactions.
* Verified and demoed real-time tracking and interactions in the browser and on Quest 3.

---

## Tech Stack

* **Data processing:** Python, PyTorch, OpenCV, Jupyter/Colab
* **Detection/Segmentation:** YOLOv8, MediaPipe, SAM, RT-DETR
* **Tracking:** SORT / DeepSORT / ByteTrack
* **Contrastive Training:** PyTorch, HuggingFace Transformers, CLIP open-source repos
* **Model Optimization:** ONNX, TensorFlow Lite, TensorFlow\.js
* **Web Integration:** TensorFlow\.js, ONNX.js, Three.js, A-Frame, Babylon.js, WebXR API, Rapier (for physics)

---

## How To Run

1. **Prepare a Dataset Subset**
   Download a subset of the Charades-Ego dataset and extract frames.

2. **Generate Annotations**
   Run detection and tracking scripts on frames to produce training data.

3. **Train Contrastive Model**
   Use the provided training script to train a MobileNetV3-based encoder.

4. **Convert/Quantize Model**
   Export the trained model to TensorFlow\.js/ONNX.js format.

5. **Start WebXR Demo**

   * Run the included web app in a browser with WebXR support (Chrome, Edge, Quest Browser).
   * Allow camera access and interact with virtual objects in real time.

---

## Demo

<!-- Insert GIFs, screenshots, or video links showing detection/tracking in WebXR, and interaction with virtual objects via Rapier. -->

---

## Roadmap / TODO

* [ ] Expand dataset for more object/body categories
* [ ] Improve caption quality and action diversity
* [ ] Try alternative contrastive losses or text encoders
* [ ] Add user controls/customization in WebXR scene
* [ ] Evaluate on additional devices (mobile, tablets, other VR headsets)

---

## References

* [Charades-Ego Dataset](https://prior.allenai.org/projects/charades-ego)
* [CLIP Paper](https://arxiv.org/abs/2103.00020)
* [MediaPipe](https://mediapipe.dev/)
* [Rapier Physics](https://rapier.rs/)
* [TensorFlow.js](https://www.tensorflow.org/js)
* [ONNX.js](https://github.com/microsoft/onnxjs)
* [SORT Tracking](https://github.com/abewley/sort)
* [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## Translating 2D Detections to 3D Space in WebXR

### Stereo Camera Approach (Two Cameras)

If you have **two cameras (stereo vision setup)**, you can obtain actual 3D coordinates of detected objects and body parts via **triangulation**. Here’s how to integrate this workflow:

#### 1. Stereo Calibration

* Use OpenCV or another vision library to calibrate both cameras.

  * Find intrinsics (focal length, optical center, distortion) and extrinsics (relative rotation and translation).
  * Use a calibration target (checkerboard) and save camera matrices and parameters.

#### 2. Frame Synchronization

* Ensure left and right camera images are paired from the same instant for each detection frame.

#### 3. Detection and Matching

* Run your detection/tracking model on both left and right camera frames.
* Match corresponding detections between the two images (using epipolar geometry or spatial proximity).

#### 4. Triangulation for 3D Reconstruction

* For each matched detection, get pixel coordinates `(u_left, v_left)` and `(u_right, v_right)`.
* Use triangulation (e.g., OpenCV’s `cv2.triangulatePoints`) with your stereo projection matrices to recover the 3D point:

```python
import cv2
import numpy as np

# P1, P2: 3x4 projection matrices for left and right camera
# pts1, pts2: Nx2 arrays of matching points (from detections)
pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3d = pts4d_hom[:3] / pts4d_hom[3]  # Convert homogeneous to (X, Y, Z)
```

#### 5. Integrating With WebXR

* Export or stream the resulting 3D positions to your browser/WebXR application.
* Use these (X, Y, Z) coordinates to place/track virtual objects or physics bodies in the XR scene.

---

### General Approaches

#### 1. If You Have Only a Single Camera (Monocular)

You’re limited: each 2D detection (bounding box or keypoint) maps to a **ray** in 3D, but not a precise position. You need more information:

* **Assume a known distance/plane:**
  Place all detections on a fixed “interaction plane” at a set distance from the camera in 3D.
  *E.g.,* If you detect a hand at `(x, y)` in 2D, project it to `(X, Y, Z)` where `Z` is a constant (e.g., 1 meter away).

* **Estimate depth heuristically:**
  Use object size (bounding box height/width) as a proxy for distance (bigger = closer, smaller = farther), or use object detection + depth estimation model.

* **Use a monocular depth estimation model:**
  Run a neural network that predicts depth maps from a single image (e.g., MiDaS, DPT, or MediaPipe Depth), then take the depth value at the detection’s center.

#### 2. If You Have Stereo Cameras or AR-Ready Devices

* **Quest 3/ARKit/ARCore/WebXR on mobile devices** can provide depth data, either via stereo cameras or SLAM.
* **Use WebXR’s Hit Test API:**
  This can map screen points (e.g., the center of your detection) to real-world 3D coordinates, if supported.

### Basic Math: 2D to 3D Projection with Known Intrinsics

If you know the camera's intrinsic parameters (focal length, principal point), you can map from 2D image points to 3D camera rays.

Given:

* Image point `(u, v)`
* Camera intrinsics (`fx`, `fy`, `cx`, `cy`)
* (Optional) depth value `Z` at that pixel

You can compute:

```python
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

This gives you `(X, Y, Z)` in camera coordinates.

### Workflow for WebXR + 2D Detection → 3D

1. **Detect object/keypoint in 2D image:**
   Get pixel `(u, v)`.

2. **Estimate depth `Z`:**

   * Fixed plane (e.g., Z = 1)
   * Or, use depth map if available.

3. **Project to 3D using camera intrinsics:**
   (as above)

4. **Transform to WebXR/world coordinates:**
   Apply any rotation/translation needed to align camera coordinates with the XR scene.

5. **Spawn/move a 3D object in the XR scene** at `(X, Y, Z)`.

### Example: Using WebXR Hit Test API

If your browser and device support it, you can do:

```js
// Get a point in normalized screen space (e.g., center of bounding box)
const xrRay = new XRRay({x: ndcX, y: ndcY, z: 0});
const hitTestResults = await xrFrame.session.requestHitTest(xrRay, referenceSpace);
// Use hitTestResults[0].getPose(referenceSpace).transform.position
```

This gives you the 3D position in the XR world for your 2D detection.

### Practical Recommendation for Your Project

* **On Quest 3 / AR devices:**
  Use WebXR’s built-in APIs to map 2D detections into real world.
  If only a single RGB stream:

  * Start by mapping to a fixed plane, or
  * Use a pre-trained depth estimation model (e.g., [MiDaS for TF.js](https://tfhub.dev/intel/midas/v2_1_small/1)) to get approximate depth.
* **For AR in the browser (WebXR):**
  The Hit Test API is the most accurate, but not available everywhere.
* **For simple VR (non-AR):**
  Map 2D detection center to a ray, choose a “default” Z, and project into the scene.

### TL;DR

* **Minimum viable:** Project detections to a fixed 3D plane in front of the camera.
* **Better:** Use a monocular depth estimator or AR depth sensor.
* **Best (if possible):** Use WebXR Hit Test API for true 3D point mapping.

---

## License

MIT (or your preferred open-source license)
