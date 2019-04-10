#!/usr/bin/env bash

if [ -d data ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir data
wget -e use_proxy=yes -e http_proxy=172.16.2.30:8080 --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=data --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd data
gunzip *
popd


mv data/t10k-images-idx3-ubyte data/test-images
mv data/t10k-labels-idx1-ubyte data/test-labels
mv data/train-images-idx3-ubyte data/train-images
mv data/train-labels-idx1-ubyte data/train-labels
