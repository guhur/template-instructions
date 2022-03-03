all: install
.PHONY: all

help :           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	
install:
	pip install poetry
	poetry install

soon:  install
	mkdir -vp data/soon/
	poetry run gdown https://drive.google.com/uc?id=1tuS3PFHOECwA5U-ofFyv6WZnTXoZjaBr -O data/soon/
	poetry run gdown https://drive.google.com/uc?id=1r6-rkaj02fdVPmFdpTlVrUX9k6WECRNb -O data/soon/
	poetry run gdown https://drive.google.com/uc?id=15Xd9DlQjWVMY-BY-jFInKF2h6syxq79e -O data/soon/
	poetry run gdown https://drive.google.com/uc?id=1zjpcTsNhsEB1frrLvncQxUusfspZqrFA -O data/soon/

reverie: install
	svn checkout https://github.com/YuankaiQi/REVERIE/trunk/tasks/REVERIE/data data/reverie
	rm -r data/reverie/.svn

datasets: soon reverie # download all datasets

template: # datasets
	poetry run extract_templates.py --mode soon --data data/soon/train.json --output data/soon/templates.json
	poetry run extract_fillers.py --data_dir data/hm3d --output data/fillers.json
	poetry run fillin_blanks.py --mode soon --tpl data/soon/templates.json --filler data/fillers.json --output data/instructions.json
	
