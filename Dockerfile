# Use the official FastAPI base image from tiangolo
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app

# Expose the port that FastAPI will run on
EXPOSE 80

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
