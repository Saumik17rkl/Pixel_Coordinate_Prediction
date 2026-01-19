# Pixel Coordinate Prediction API

A FastAPI-based web service that predicts pixel coordinates from input images using a deep learning model.

## Features

- Accepts image uploads and returns predicted (x, y) coordinates
- Built with FastAPI for high performance
- Preprocesses images to enhance signal and reduce noise
- Includes Swagger UI documentation at `/docs`

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ImageProcessing
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your model file (`pixel_coordinate_regressor.keras`) in the project root directory.

## Usage

### Running Locally

Start the development server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Accepts an image file and returns predicted coordinates

### Example Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_image.png;type=image/png'
```

### Example Response

```json
{
  "predicted_x": 25,
  "predicted_y": 37
}
```

## Deployment

### Render

This application includes a `render.yaml` configuration file for easy deployment to Render:

1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)
2. Connect your repository to Render
3. Select the repository and let Render handle the deployment

### Environment Variables

- `PORT`: Port number to run the server on (default: 8000)
- `MODEL_PATH`: Path to the model file (default: "pixel_coordinate_regressor.keras")

## Development

### Testing

To test the API locally, you can use the interactive documentation at `http://127.0.0.1:8000/docs` or tools like [Postman](https://www.postman.com/) or [curl](https://curl.se/).

## Model Information

The model expects input images of size 50x50 pixels and outputs a heatmap from which the predicted coordinates are extracted.

## License

[Specify your license here]# Pixel_Coordinate_Prediction
# Pixel_Coordinate_Prediction
