# Image Recommendation System Using Vector Search and Embeddings

## Overview
This repository provides an open-source implementation of an image recommendation system using vector search and deep learning-based image embeddings. By leveraging pre-trained models for feature extraction and vector search libraries for similarity queries, this system is designed to recommend images based on visual similarity.

## Features
- **Image Embedding Extraction**: Uses pre-trained convolutional neural networks (e.g., InceptionV3) to extract feature vectors from images.
- **Vector Search**: Employs FAISS (Facebook AI Similarity Search) for high-speed similarity search across high-dimensional embeddings.
- **API Interface**: Includes a REST API built with FastAPI for easy querying and integration.
- **Scalable Solution**: Handles large-scale image datasets efficiently with support for real-time recommendations.

## Requirements
- Python 3.8+
- TensorFlow or PyTorch (for model loading)
- FAISS for vector search
- FastAPI (or Flask for alternative API setup)
- NumPy
- OpenCV (optional for image processing)

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/image-recommendation-system.git
   cd image-recommendation-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup
1. **Prepare Image Embeddings**:
   - Ensure your image dataset is in a folder (e.g., `data/images/`).
   - Run the feature extraction script to generate embeddings:
     ```bash
     python extract_embeddings.py --image_folder data/images/ --output embeddings.npy
     ```

2. **Build the FAISS Index**:
   ```bash
   python build_index.py --embedding_path embeddings.npy --output_path faiss_index.bin
   ```

3. **Start the API Server**:
   ```bash
   uvicorn main:app --reload
   ```

## Usage
1. **Query via API**:
   Send a POST request to the `/recommend/` endpoint with an image file.
   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/recommend/' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@path/to/your/query_image.jpg'
   ```

2. **Response**:
   The server will return a JSON object with indices of the recommended images and their similarity scores.

## Code Overview
- **extract_embeddings.py**: Loads images, processes them using a pre-trained model, and saves their embeddings.
- **build_index.py**: Constructs a FAISS index using the extracted embeddings.
- **main.py**: API code using FastAPI to serve recommendations.
- **utils.py**: Contains helper functions for image preprocessing and embedding extraction.

## Evaluation
The system's performance can be evaluated using metrics such as precision at K (P@K) and mean average precision (mAP). Example scripts for evaluation are included in the `evaluation/` folder.

## Future Work
- Integrate multi-modal embeddings (e.g., combining image and text).
- Extend support for more advanced deep learning models.
- Improve user personalization with metadata or collaborative filtering.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or raise issues with suggestions and improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **TensorFlow/Keras**: For model implementation and feature extraction.
- **FAISS**: For providing a robust vector search library.
- **FastAPI**: For making API development straightforward and efficient.
