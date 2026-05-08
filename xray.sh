docker run --rm -ti -p 7860:7860 -v ./root:/root/ -v ./python3.12/:/usr/local/lib/python3.12/ -e NO_ALBUMENTATIONS_UPDATE=1 python:3.12 bash

