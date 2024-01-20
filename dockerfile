# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster as builder

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install project dependencies.
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Now copy over all the poetry installed packages to our
# actual final image
FROM python:3.10-slim-buster

WORKDIR /app
COPY --from=builder /usr/local /usr/local

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the command to start uWSGI
CMD ["uvicorn", "fastapi_serving:app", "--host", "0.0.0.0", "--port", "80"]