# Pixel Coordinate Prediction API

A Flask-based web service that predicts pixel coordinates from input images using a deep learning model.

## Features

- Accepts image uploads and returns predicted (x, y) coordinates
- Built with Flask for high performance
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

4. Set up environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file with your configuration
   # (The file is in .gitignore for security)
   ```

5. Place your model file (`pixel_coordinate_regressor.keras`) in the project root directory or update `MODEL_PATH` in your `.env` file.

## Usage

### Running Locally

Start the development server:
```bash
flask run --host=0.0.0.0 --port=5000
```

The API will be available at `http://127.0.0.1:5000`

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Accepts an image file and returns predicted coordinates

Example request using `curl`:

```bash
curl -X POST -F 'file=@path_to_your_image.png' http://localhost:5000/predict
```

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

Create a `.env` file in the project root with the following variables:

```
# Server Configuration
PORT=8000
HOST=0.0.0.0

# Application Settings
MODEL_PATH=pixel_coordinate_regressor.keras
IMAGE_SIZE=50

# Environment (development/production)
ENV=development

# CORS Settings (for development, restrict in production)
ALLOWED_ORIGINS=*
```

You can copy `.env.example` to `.env` as a starting point.

## Development

### Testing

To test the API locally, you can use tools like [Postman](https://www.postman.com/) or [curl](https://curl.se/).

## Model Information

The model expects input images of size 50x50 pixels and outputs a heatmap from which the predicted coordinates are extracted.

## License

[Specify your license here]# Pixel_Coordinate_Prediction
# Pixel_Coordinate_Prediction
