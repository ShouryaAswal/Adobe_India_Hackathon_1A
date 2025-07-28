# Stage 1: Base Image
# Use a specific version of python for reproducibility and a slim version for size.
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching.
# This step is isolated so that changes to the source code don't invalidate
# the installed packages layer.
COPY requirements.txt .

# Install Python dependencies using the copied requirements file.
# The --no-cache-dir flag is used to reduce the final image size.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


# Copy the rest of the application code and the model into the container.
# This includes your src/ folder and your trained model/ folder.
COPY src/ ./src
COPY model/ ./model

# Define the default command to run when the container starts.
# This executes your main Python script to start the processing pipeline.
CMD ["python", "src/main.py"]
