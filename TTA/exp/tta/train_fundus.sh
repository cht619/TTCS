GPUS=$1
dataset=$2

# 数量从小到大： Drishti_GS, RIM_ONE_r3,ORIGA, REFUGE
#if [ $dataset == 0 ];then
#  Data=('Drishti_GS' 'ORIGA' 'RIM_ONE_r3')

# 自己定义
if [ $dataset == 0 ];then
  Data=('RIM_ONE_r3' 'ORIGA')
elif [ $dataset == 1 ]; then
    Data=('Drishti_GS' 'REFUGE' 'REFUGE_Valid')
elif [ $dataset == 2 ]; then
  Data=('RIM_ONE_r3' 'REFUGE' 'ORIGA' 'REFUGE_Valid' 'Drishti_GS')
elif [ $dataset == 5 ]; then
  Data=('Drishti_GS' 'Drishti_GS' 'REFUGE_Valid')
else
  Data=($dataset)
fi

# nums: 159, 400, 650, 800, 101
All_Data=('RIM_ONE_r3' 'REFUGE' 'ORIGA' 'REFUGE_Valid' 'Drishti_GS')  # ABCDE的顺序


config='configs/tta/tta_Fundus.py'

echo ${Data[*]}

for data in ${Data[@]}
do
      export CUDA_VISIBLE_DEVICES=$GPUS && export HF_ENDPOINT=https://hf-mirror.com &&  CUDA_LAUNCH_BLOCKING=1 && \
      python TTA/main.py \
        --config=$config \
        --domain=$data \
        --baseline=$3
done


# bash TTA/exp/tta/train_fundus.sh 0 RIM_ONE_r3 SAMCLIP_TTA
# bash TTA/exp/tta/train_fundus.sh 1 REFUGE SAMCLIP_TTA
# bash TTA/exp/tta/train_fundus.sh 2 ORIGA SAMCLIP_TTA
# bash TTA/exp/tta/train_fundus.sh 3 REFUGE_Valid SAMCLIP_TTA
# bash TTA/exp/tta/train_fundus.sh 4 Drishti_GS SAMCLIP_TTA