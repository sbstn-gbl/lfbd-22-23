VIRTUALENV?=env
PORT?=8888

help:
	@echo "Make targets:"
	@echo "  build          create virtualenv and install packages"
	@echo "  build-lab      `build` + lab extensions"
	@echo "  freeze         persist installed packaged to requirements.txt"
	@echo "  clean          remove *.pyc files and __pycache__ directory"
	@echo "  distclean      remove virtual environment"
	@echo "  run            run jupyter lab (default port $(PORT))"
	@echo "Check the Makefile for more details"

build:
	virtualenv $(VIRTUALENV); \
	source $(VIRTUALENV)/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt;

build-lab: build
	source $(VIRTUALENV)/bin/activate; \
	jupyter serverextension enable --py jupyterlab_code_formatter

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d | xargs rm -fr
	find . -name '.ipynb_checkpoints' -type d | xargs rm -fr

distclean: clean
	rm -rf $(VIRTUALENV)

run:
	source $(VIRTUALENV)/bin/activate; \
	jupyter lab --no-browser --port=$(PORT)
