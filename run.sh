#!/bin/bash


echo "start server..."

python server.py &

sleep 3

echo "start clients..."

for i in {0..4}
do
	python client.py $i &
done

wait

