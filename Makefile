env:
	conda env create -f env.yml

split:
	python scripts/split.py

profiles:
	python scripts/profiles.py

transform-y:
	python scripts/transform_y.py

variance:
	python scripts/variance.py

skew:
	python scripts/skew.py