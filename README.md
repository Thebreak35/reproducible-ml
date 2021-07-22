# Reproducible Machine learning

From [Reproducible Deep Learning](https://www.sscardapane.it/teaching/reproducibledl/)

## Prerequisite
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)
- [Anaconda](https://www.anaconda.com/products/individual)

## Installation

1. Clone repository 
```
git clone https://github.com/Thebreak35/reproducible-ml.git
```
2. Create conda environment from `environment.yaml`, using this below command.
```
conda env create -f environment.yml --name dvc
```
3. Activate the environment.
```
conda activate dvc
```

## Usage

After activate dvc environment.

1. Add dvc remote.
```
dvc remote add --local -d my-remote ${DVC_STORAGE_TYPE}://${DVC_STORAGE_PATH}
```
`--local` is any remote configurations are not tracked by Git.
`-d` is default remote.
`DVC_STORAGE_TYPE` such as s3, gdrive, ssh, webhdfs or local.
`DVC_STORAGE_PATH` path to data.

Documentation [remote add](DVC_STORAGE_USER) 

2. Pull dataset and base models from the storage.
```
dvc pull
```

3. Build image from Dockerfile
```
docker build -t sefr:image .
```

4. Run container from `sefr:image` image.
```
docker container run -d --rm -v $(pwd):/train --name sefr-container sefr:image
```