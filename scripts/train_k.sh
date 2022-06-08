#!/bin/bash
ds=$1
if [ -z "$ds" ];then
	echo "./run.sh 20ng/tmn/webs/reuters/imdb/wikitext"
	exit 0
fi
mkdir -p run-logs
tau='0.7'
K_list='20 30 50 75 100'
for K in $K_list; do
	TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
	echo "[$TIMESTAMP]=============== train ${ds}:${K}topics:cl=1 tau=${tau} Start ==============="
	python train.py $ds $K $tau >> run-logs/$ds-$K.log 2>&1
	TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
	echo "[$TIMESTAMP]=============== train ${ds}:${K}topics:cl=1 tau=${tau} End ==============="
done

