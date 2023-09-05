# Use a Python base image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Command to run the training script
CMD ["python", "train.py"]
