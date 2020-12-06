# Joint dialogue act segmentation and recognition

Most of this code is based on Tianyu Zhao's repo:
https://github.com/ZHAOTING/dialog-processing

I copied relevant parts from src/tasks/joint_da_seg_recog

Some preprocessing code is based on Hao Cheng's repo:
https://github.com/hao-cheng/dynamic_speaker_model

## Steps


## TODO items/issues
Priority:
- [x] Implement sequence tagging model 
- [x] Change LR scheduling schemes 
- [ ] Add ASR experiments

Longer-term:
- [ ] Fix `run_test` and overhaul config ... to not have to pass all the params again during testing
- [ ] Consolidate all models into one module; though needs some more thought as the tokenization is different

