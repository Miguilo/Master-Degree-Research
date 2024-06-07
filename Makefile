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

generate_performance_images:
	cd ../general/
	poetry run python save_performances_img

generate_FI_images:
	cd ./src_apolar/
	poetry run python feature_importance_apolar.py

	cd ../src_polar/
	poetry run python feature_importance_polar.py

	cd ../src_polar_apolar/
	poetry run python feature_importance_polar_apolar.py


