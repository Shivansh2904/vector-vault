.PHONY: install backend frontend test docker-up docker-down clean

PYTHON ?= python
PIP ?= pip

install: install-backend install-frontend

install-backend:
	cd backend && $(PIP) install -r requirements.txt

install-frontend:
	cd frontend && npm install

backend:
	cd backend && uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && $(PIP) install pytest httpx && pytest tests/ -v

build-frontend:
	cd frontend && npm run build

docker-up:
	docker compose up

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist frontend/node_modules

help:
	@echo "Common targets:"
	@echo "  make install        Install backend + frontend deps"
	@echo "  make backend        Run FastAPI server on :8000"
	@echo "  make frontend       Run Vite dev server on :5173"
	@echo "  make test           Run backend tests"
	@echo "  make docker-up      Run via docker compose"
