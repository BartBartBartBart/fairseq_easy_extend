#!/bin/bash

python3 decode_interactive.py drive/fairseq-data-roen \
--task translation_lev \
--iter-decode-max-iter 9 \
--iter-decode-eos-penalty 0 \
--path drive/cmlm_checkpoint_best.pt \
--input drive/newstest2016.tok.tc.spm.ro \
--temp 1.3 \
