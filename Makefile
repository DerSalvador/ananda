run:
	make -C app run

requirements:
	make -C app requirements

copy:
	cp docker-compose.yml ftrade/docker-compose.yml
	cp AnandaStrategySplit.py ftrade/user_data/strategies/AnandaStrategySplit.py

test: copy
	cd ftrade; docker compose up
