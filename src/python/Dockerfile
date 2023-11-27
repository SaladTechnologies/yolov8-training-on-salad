# Use the official Python image as the base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the inference folder to /app/inference
COPY training /app/training

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Update pip and install requirements
RUN pip install --upgrade pip
RUN pip install  -r training/requirements.txt

# Create a folder for checkpoints
RUN mkdir /app/training/runs

# Change mode to read/write/execute for everyone
RUN chmod -R 777 /app/training/runs

# Make sync.sh executable
RUN chmod +x /app/training/sync.sh

# Download AzCopy
RUN wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy.tar.gz --strip-components=1

# Move AzCopy to the /usr/bin directory
RUN mv azcopy /usr/bin/

# Move to the code repo
WORKDIR /app/training

# Set the command to run when the Docker container starts
CMD ["./sync.sh"]