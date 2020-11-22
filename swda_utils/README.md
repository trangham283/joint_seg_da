# SWDA Preprocessing Scripts

Code based on:

* [Tianyu Zhao's dialog processing repo, swda preprocessing step](https://github.com/ZHAOTING/dialog-processing/tree/master/src/tasks/joint_da_seg_recog)
* [Hao Cheng's dynamic speaker model repo, swda preprocessing step](https://github.com/hao-cheng/dynamic_speaker_model/blob/master/data_script/process_predictor_data.py)

## Steps:
* Set the right data paths in `config.py`
* Run `build_joint_da_seg_recog_dataset.py`
* Run `align_times.py`
* Run `processed_aligned.py`
* Run `get_speech_features.py`
