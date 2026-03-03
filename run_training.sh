#!/bin/bash


accelerate launch --config_file default_config.yaml train_script.py -c train_configs/train_example.yml

