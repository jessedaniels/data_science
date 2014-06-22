#!/bin/bash

python vw_wholesale_cust.py | shuf | vw -k -c --passes 30 --oaa 2 --l2 0.001 --loss_function logistic --holdout_off
