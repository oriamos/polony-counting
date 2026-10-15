.DEFAULT_GOAL := help
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Polony Review app (src/polony_app + frontend/, see docs/app/README.md)

# The app lives in its own virtualenv (.venv, git-ignored). Use your own interpreter with
# `make app APP_PY=python` (then dependencies are installed into that interpreter).
APP_VENV ?= .venv
APP_PY ?= $(APP_VENV)/bin/python
APP_DATA_DIR ?= ./data
PORT ?= 8000

.PHONY: app-install
app-install: ## create .venv (if APP_PY is the default) and install requirements-app.txt
	@if [ "$(APP_PY)" = "$(APP_VENV)/bin/python" ] && [ ! -x "$(APP_PY)" ]; then python3 -m venv $(APP_VENV); fi
	@$(APP_PY) -m pip install --quiet -r requirements-app.txt
	@echo "app dependencies installed for $(APP_PY)"

.PHONY: app-run
app-run: ## build frontend (if npm is present) and run the app on http://127.0.0.1:8000 (PORT=...)
	@test -x "$$(command -v $(APP_PY))" || { echo "$(APP_PY) not found: run 'make app-install' first"; exit 1; }
	@APP_PY=$(APP_PY) POLONY_DATA_DIR=$(APP_DATA_DIR) PORT=$(PORT) ./scripts/run_app.sh

.PHONY: app-test
app-test: ## run the backend test-suite (tests/app)
	@test -x "$$(command -v $(APP_PY))" || { echo "$(APP_PY) not found: run 'make app-install' first"; exit 1; }
	@PYTHONPATH=src $(APP_PY) -m pytest tests/app -q

.PHONY: web-test
web-test: ## frontend typecheck + unit tests + mock end-to-end
	@cd frontend && npm install --no-audit --no-fund && npx tsc --noEmit && npx vitest run && npm run build && npm run e2e

.PHONY: test
test: app-test web-test ## backend + frontend tests

.PHONY: app
app: app-install app-run ## ONE command: install (venv) + build frontend (if npm) + run

##@ Installer (install/, see docs/app/INSTALL.md)

.PHONY: web
web: ## rebuild the frontend and copy it to src/polony_app/webdist (committed; the one-command installer serves it)
	@cd frontend && npm install --no-audit --no-fund && npm run build
	@rm -rf src/polony_app/webdist && mkdir -p src/polony_app/webdist && cp -R frontend/dist/. src/polony_app/webdist/
	@echo "webdist rebuilt: commit src/polony_app/webdist together with the frontend change"

.PHONY: package-icons
package-icons: ## regenerate install/assets/PolonyReview.{png,ico,icns} (needs Pillow)
	@$(APP_PY) install/make_icons.py

##@ Clean-up

.PHONY: clean
clean: ## remove caches, builds and reports
	@rm -rf .pytest_cache htmlcov .coverage coverage.xml frontend/dist
	@find . -name __pycache__ -not -path './.venv/*' -not -path '*/node_modules/*' -prune -exec rm -rf {} +
