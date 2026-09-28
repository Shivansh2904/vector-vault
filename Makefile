.PHONY: help install install-backend install-frontend backend frontend test build-frontend docker-up docker-down clean

PYTHON ?= python

help:
	@echo "make install         Install backend (with test tools) and frontend dependencies"
	@echo "make backend         Run the API on http://localhost:8000"
	@echo "make frontend        Run the Vite dev server on http://localhost:5173"
	@echo "make test            Run the backend tests"
	@echo "make build-frontend  Type-check and build the frontend"
	@echo "make docker-up       Build and start both services with Docker Compose"
	@echo "make docker-down     Stop them"
	@echo "make clean           Remove Python caches, frontend/dist and frontend/node_modules"

install: install-backend install-frontend

install-backend:
	cd backend && $(PYTHON) -m pip install -r requirements-dev.txt

install-frontend:
	cd frontend && npm install

backend:
	cd backend && $(PYTHON) -m uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && $(PYTHON) -m pytest tests/ -v

build-frontend:
	cd frontend && npm run build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find backend examples -path backend/.venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache frontend/dist frontend/node_modules
