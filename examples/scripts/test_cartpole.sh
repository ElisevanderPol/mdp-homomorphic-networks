#!/bin/bash

# Run 25 seeds with the best-performing learning rate for each network
for i in {0..24}
do
	python cartpole_ppo.py --cuda_idx 0 --run_ID $i --network equivariant --lr 0.01 --fcs 64 64 &
	python cartpole_ppo.py --cuda_idx 0 --run_ID $i --network nullspace --lr 0.005 --fcs 64 64 &
	python cartpole_ppo.py --cuda_idx 0 --run_ID $i --network random --lr 0.001 --fcs 64 64 &
	python cartpole_ppo.py --cuda_idx 0 --run_ID $i --network mlp --lr 0.001 --fcs 64 128 &
	wait
done
