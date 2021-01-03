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

corr:
	python scripts/06_corr.py

pca:
	python scripts/07_pca.py

ranking-raw:
	python scripts/08_ranking_raw.py

profile-final:
	python scripts/09_profile_final.py

ranking-final:
	python scripts/10_ranking_final.py

feature-selection:
	python scripts/11_feature_selection.py

features:
	python scripts/12_features.py

fits:
	python scripts/13_fits.py