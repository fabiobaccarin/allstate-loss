env:
	conda env create -f env.yml

split:
	python scripts/01_split.py

profiles:
	python scripts/02_profiles.py

transform-y:
	python scripts/03_transform_y.py

variance:
	python scripts/04_variance.py

skew:
	python scripts/05_skew.py

pca:
	python scripts/06_pca.py