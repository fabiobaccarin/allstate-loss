env:
	conda env create -f env.yml

split:
	python scripts/split.py

profiles:
	python scripts/profiles.py