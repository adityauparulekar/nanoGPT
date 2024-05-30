NAME=$1
NUM=$2
P=$3
ITER=$4
RESUM=$5
echo $NAME

cd data/$NAME
python generate.py $NUM $P
cd ../..
python data/prepare_generator.py $NAME
# echo "GENERATED PREPARE"
python data/$NAME/prepare.py
# echo "RAN PREPARE"
python config/config_generator.py $NAME $ITER $RESUME
# echo "GENERATED CONFIG"
python train.py config/$NAME.py --name=$NAME
# echo "TRAINED MODEL"
rm config/$NAME.py
python sample.py --num=$NUM --p=$P --iter=$ITER --out_dir=out-$NAME
# python output_judge_history.py $NUM $P $ITER