.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

docs_view:
	@echo View API documentation... 
	pdoc src --http localhost:8080

docs:
	@echo Save documentation to docs... 
	pdoc src -o docs --html

generate_images:
	