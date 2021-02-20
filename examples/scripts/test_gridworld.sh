#!/bin/bash

# Run 20 seeds with the best-performing learning rate for each network
for i in {0..19}
do
	python grid_a2c.py --cuda_idx 0 --run_ID $i --network equivariant --lr 0.001 &
	python grid_a2c.py --cuda_idx 0 --run_ID $i --network nullspace --lr 0.003 &
	python grid_a2c.py --cuda_idx 0 --run_ID $i --network random --lr 0.001 &
	python grid_a2c.py --cuda_idx 0 --run_ID $i --network cnn --lr 0.003 &
	wait
done

