venv:
	python3 -m venv env
	@echo Run the following command
	@echo source env/bin/activate

requirements:
	source env/bin/activate; python3.10 -m pip install -r requirements.txt

run:
	source env/bin/activate; python3.10 main.py

freeze:
	source env/bin/activate; python3.10 -m pip freeze
