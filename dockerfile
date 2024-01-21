# Use an official Python runtime as a parent image
FROM python:3.11.5 as builder

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install project dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Now copy over all the pip installed packages to our
# actual final image
FROM python:3.11.5

WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Define the command to start the container
CMD ["uvicorn", "src.fastapi_serving:app", "--host", "0.0.0.0", "--port", "8000"]