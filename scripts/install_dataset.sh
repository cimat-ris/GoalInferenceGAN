#!/usr/bin/env bash
# Will download ~80MB file from: https://drive.google.com/open?id=18AvK8bUhda7rOhW0EDysLxW6ifkVFC6w
FILEID=18AvK8bUhda7rOhW0EDysLxW6ifkVFC6w
FILENAME="ewap_dataset_full.zip"
if [ -f "$FILENAME" ]; then
    echo "$FILENAME already downloaded"
else
    echo "$FILENAME downloading..."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
fi

# unzip downloaded file to  destination
DATASET_DIR="samples/datasets/"
unzip -u $FILENAME -d $DATASET_DIR

# And now remove downloaded file
rm $FILENAME