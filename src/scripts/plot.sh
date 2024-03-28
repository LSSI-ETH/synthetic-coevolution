#!/bin/bash

path="synthetic-coevolution/src/plot/"
cd $path
python process_results.py
python plot_performance.py

