run-engine:
	cd apps/engine-go && go run .

run-brain:
	cd apps/brain-rl && venv/Scripts/python api.py

run-dash:
	cd apps/dashboard-js && npm run dev

setup:
	@echo "Setting up Go..."
	cd apps/engine-go && go mod tidy
	@echo "Setting up Python..."
	cd apps/brain-rl && pip install -r requirements.txt