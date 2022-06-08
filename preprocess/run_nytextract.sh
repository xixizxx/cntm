#!/bin/bash
b=1987
#e=2007
e=1987
while [ $b -le $e ]
do
	python nytextract.py -o ../../data/nyt_orig/${b}.csv -d ../.. ../../data/nyt_orig/data/${b}
	((b++))
done
