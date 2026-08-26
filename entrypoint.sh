#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [ "$DJANGO_ENV" != "production" ] && [ -n "$DB_HOST" ]; then
  echo "Waiting for database..."
  DB_PORT=${DB_PORT:-5432}

  if command -v nc >/dev/null 2>&1; then
    while ! nc -z $DB_HOST $DB_PORT; do
      echo "Database not ready, waiting..."
      sleep 1
    done
  else
    echo "nc not found, sleeping for 5s..."
    sleep 5
  fi

  echo "Database is ready!"
fi

# Run migrations unless SKIP_MIGRATIONS is set
if [ "$SKIP_MIGRATIONS" != "true" ]; then
  echo "Running migrations..."
  uv run python manage.py migrate --noinput
fi

if [ "$DJANGO_ENV" = "production" ]; then
  echo "Running in PRODUCTION mode"
  
  echo "Collecting static files..."
  uv run python manage.py collectstatic --noinput

  if [ $# -gt 0 ]; then
    echo "Running custom production command: $@"
    exec uv run "$@"
  else
    echo "Starting Gunicorn..."
    exec uv run gunicorn src.wsgi:application \
        --bind 0.0.0.0:${PORT:-8000} \
        --workers 1 \
        --threads 2 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        --log-level info
  fi
else
  echo "Running in DEVELOPMENT mode"
  
  if [ $# -gt 0 ]; then
    echo "Running custom development command: $@"
    exec uv run "$@"
  else
    echo "Starting Django development server..."
    exec uv run python manage.py runserver 0.0.0.0:8000
  fi
fi
