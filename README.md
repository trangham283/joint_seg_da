# Joint dialogue act segmentation and recognition

## Acknowledgement
* Most of this code is based on Tianyu Zhao's [dialog processing repo](https://github.com/ZHAOTING/dialog-processing); I copied relevant parts from `src/tasks/joint_da_seg_recog`
* Some SWDA preprocessing code is based on Hao Cheng's [dynamic speaker repo](https://github.com/hao-cheng/dynamic_speaker_model)
* Learning/optimizer common values, general tips, and good coding practices (though can't say I've successfully adopted them all) are based on Nikita Kitaev's [self-attentive parser repo](https://github.com/nikitakit/self-attentive-parser), as well as Nils Reimers' [sentence transformers repo](https://github.com/UKPLab/sentence-transformers). These are beautiful codebases and I learned so much from just studying the code.

## Steps


## TODO items/issues
Priority:
- [x] Implement sequence tagging model 
- [x] Change LR scheduling schemes 
- [ ] Add ASR experiments

Longer-term:
- [ ] Fix `run_test` and overhaul config ... to not have to pass all the params again during testing
- [ ] Consolidate all models into one module; though needs some more thought as the tokenization is different

