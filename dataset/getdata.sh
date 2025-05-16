#!/bin/bash

mkdir -p data images
cd data
wget https://figshare.com/ndownloader/articles/22004273/versions/2 -O 22004273.zip
unzip 22004273.zip
cd ../images

start_url="https://spatial-dlpfc.s3.us-east-2.amazonaws.com/images"

filenames="151676 151669 151507 151508 151672 151670 151673 151675 151510 151671 151674 151509"
for id in $filenames; do
    url="${start_url}/${id}_full_image.tif"
    wget "$url" -O "${id}.tif"
done
cd -
