if [[ $# -eq 0 ]] ; then
    echo 'SRC_DIR is missing'
    exit 0
fi

SRC_DIR=$1

# delete empty files
rm ${SRC_DIR}/Cat/666.jpg
rm ${SRC_DIR}/Dog/11702.jpg

mkdir -p ~/data/train

mv ${SRC_DIR}/Cat ~/data/train/cats
mkdir -p ~/data/validation/cats
mv ~/data/train/cats/*0.jpg ~/data/validation/cats

mv ${SRC_DIR}/Dog ~/data/train/dogs
mkdir -p ~/data/validation/dogs
mv ~/data/train/dogs/*0.jpg ~/data/validation/dogs
