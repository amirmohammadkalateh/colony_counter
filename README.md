# colony_counter

# Colony Counter with TensorFlow and Roboflow

An advanced bacterial colony counting system using deep learning and computer vision, built with TensorFlow and integrated with Roboflow for dataset management.

## 🔑 Key Features

- Automated colony counting using deep convolutional neural networks
- Integration with Roboflow for dataset management and versioning
- Real-time image preprocessing and analysis
- Support for various image formats and sizes
- Customizable model architecture
- Easy-to-use prediction interface

## 🛠️ Technical Architecture

The system consists of several key components:

1. **Image Preprocessing**: Uses OpenCV to standardize input images to 224x224 pixels
2. **Deep Learning Model**: Custom CNN architecture with:
   - Multiple convolutional layers with ReLU activation
   - MaxPooling layers for feature extraction
   - Dense layers for final count prediction
3. **Roboflow Integration**: Handles dataset versioning and management
4. **TensorFlow Backend**: Powers the deep learning capabilities

## 📋 Requirements

The project requires the following dependencies (automatically installed via pyproject.toml):
- TensorFlow (>=2.13.0)
- OpenCV Python (>=4.8.0)
- Roboflow (>=1.1.5)
- NumPy (<2.0.0)
- Matplotlib (>=3.7.1)

## 🚀 Getting Started

1. Set up your Roboflow API key in the Replit Secrets:
   - Click on the "Tools" button
   - Select "Secrets"
   - Add a new secret with key `ROBOFLOW_API_KEY`
   - Paste your Roboflow API key as the value

2. Update the workspace name in `main.py`:
   ```python
   project = rf.workspace("your-workspace-name").project("colony-counter-0ltmm")
   ```

3. Click the "Run" button to start the application

## 🎯 Model Architecture

The neural network architecture is designed specifically for colony counting:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

## 📈 Training Process

The model is trained using:
- Mean Squared Error (MSE) as the loss function
- Adam optimizer
- Mean Absolute Error (MAE) as a metric
- 10 epochs by default
- Automatic validation split

## 🔄 Workflow

1. Images are preprocessed and normalized
2. The model predicts colony counts
3. Results are returned as integer values
4. Optional visualization tools are available

## 🐛 Troubleshooting

Common issues and solutions:

1. **TensorFlow Warnings**: The CPU optimization warnings can be safely ignored
2. **Roboflow Authentication**: Ensure your API key is properly set in Secrets
3. **Memory Issues**: Consider reducing batch size if encountering memory problems

## 🤝 Contributing

Feel free to fork this project and submit improvements. Key areas for potential enhancement:

- Additional data augmentation techniques
- Model architecture improvements
- UI/UX enhancements
- Performance optimizations


## 📫 Support

For issues and questions:
1. Open an issue in the project repository
2. Contact the project maintainers
3. Check the documentation for updates

---

Made with ❤️ using TensorFlow and Roboflow
