# Use the official Python image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies (for OpenCV and Matplotlib)
RUN apt-get update && apt-get install -y libgl1-mesa-glx fontconfig  # For Ubuntu/Debian

# Set the MPLCONFIGDIR environment variable for Matplotlib cache directory
ENV MPLCONFIGDIR=/tmp/matplotlib

# Set the OSFONTDIR environment variable for Fontconfig cache directory
ENV OSFONTDIR=/usr/share/fonts

# Run fc-cache with appropriate permissions to create Fontconfig cache
RUN chmod -R o+w /usr/share/fonts \
  && fc-cache --really-force --verbose

# Create a directory for Numba cache with appropriate permissions
RUN mkdir -p /app/numba_cache && chmod 777 /app/numba_cache

# Copy the requirements file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (from requirements.txt)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire current directory into the container at /app
COPY ./app /code/app

#copy the output file for fer library
COPY ./output /code/output

# Set the environment variable for the model file path and Numba cache directory
ENV NUMBA_CACHE_DIR=/app/numba_cache

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
