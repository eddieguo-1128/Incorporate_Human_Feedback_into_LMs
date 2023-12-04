#!/bin/bash


curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

git lfs install

git clone https://huggingface.co/weqweasdas/hh_rlhf_rm_open_llama_3b

mv hh_rlhf_rm_open_llama_3b hh-rlhf-rm-open-llama-3b
