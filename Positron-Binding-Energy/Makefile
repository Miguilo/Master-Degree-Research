.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install
	cp transformers.py /home/$(USER)/.cache/pypoetry/virtualenvs/pbe_for_molecules-GBFEFQnP-py3.9/lib/python3.9/site-packages/skopt/space/transformers.py
	poetry run ipython kernel install --user --name=pbe_for_molecules

activate:
	@echo "Activating virtual environment"
	poetry shell

docs_view:
	@echo View API documentation... 
	pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	pdoc src -o docs

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache