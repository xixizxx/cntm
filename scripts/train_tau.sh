#!/bin/bash
ds=$1
if [ -z "$ds" ];then
	echo "./run.sh 20ng/tmn/webs/reuters/imdb/wikitext"
	exit 0
fi
mkdir -p run-logs
tau_list='0 0.05 0.2 0.5 0.7 1'
for tau in $tau_list; do
	TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
	echo "[$TIMESTAMP]=============== train ${ds}:50topics:cl=1 tau=$tau Start ==============="
	python train.py $ds 50 $tau >> run-logs/$ds-$tau.log 2>&1
	TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
	echo "[$TIMESTAMP]=============== train ${ds}:50topics:cl=1 tau=$tau End ==============="
done

