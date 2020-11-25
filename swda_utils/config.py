class SpeechConfig(object):
    def __init__(self):
        self.dialog_acts = ['sd', 'b', 'sv', '%', 'aa', 'ba', 'qy', 'ny', 'fc',
                'qw', 'nn', 'bk', 'fo_o_fw_"_by_bc', 'h', 'qy^d', 'bh', '^q', 
                'bf', 'na', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 
                'br', 'no', 'fp', 'qrr', 'arp_nd', 't3', 'oo_co_cc', 't1', 'bd',
                'aap_am', '^g', 'qw^d', 'fa', 'ft']
        self.joint_da_seg_recog_labels = ["I"] + ["E_"+da for da in self.dialog_acts]
        self.n_workers = 1
        
        # Management
        self.raw_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/swda"
        self.task_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/joint"
        self.dataset_path = f"{self.task_data_dir}"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"
        self.feature_dir = "/s0/ttmt001/acoustic_features_json"
        self.cache_dir = "/s0/ttmt001"
        self.model_save_path = "/s0/ttmt001/joint_da"
        self.suffix = "_bert_time_data.json"
        self.pause_vocab = {"<PAD>": 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 
                "<START>": 7, "<STOP>": 8}
        self.feat_sizes = {'pitch': 3, 'fb3': 3, 'mfcc': 13, 'fbank': 41}

        # data splits
        self.train_dialog_list = f"{self.task_data_dir}/train.txt"
        self.test_dialog_list = f"{self.task_data_dir}/test.txt" 
        self.dev_dialog_list = f"{self.task_data_dir}/dev.txt" 

        # other
        self.ms_dir = "/g/ssli/data/treebank/ms_alignment/swb_ms98_transcriptions/"
        #self.out_dir = "/s0/ttmt001/swda_tsv"
        self.out_dir = self.task_data_dir

class TrainConfig(object):
    def __init__(self):
        self.dialog_acts = ['sd', 'b', 'sv', '%', 'aa', 'ba', 'qy', 'ny', 'fc',
                'qw', 'nn', 'bk', 'fo_o_fw_"_by_bc', 'h', 'qy^d', 'bh', '^q', 
                'bf', 'na', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 
                'br', 'no', 'fp', 'qrr', 'arp_nd', 't3', 'oo_co_cc', 't1', 'bd',
                'aap_am', '^g', 'qw^d', 'fa', 'ft']
        self.joint_da_seg_recog_labels = ["I"] + ["E_"+da for da in self.dialog_acts]
        self.n_workers = 1
        
        # Management
        self.raw_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/swda"
        self.task_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/joint"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"
        self.cache_dir = "/s0/ttmt001"
        self.model_save_path = "/s0/ttmt001/joint_da"

        # data splits
        self.train_dialog_list = f"{self.task_data_dir}/train.txt"
        self.test_dialog_list = f"{self.task_data_dir}/test.txt" 
        self.dev_dialog_list = f"{self.task_data_dir}/dev.txt" 

        # other
        self.ms_dir = "/g/ssli/data/treebank/ms_alignment/swb_ms98_transcriptions/"
        #self.out_dir = "/s0/ttmt001/swda_tsv"
        self.out_dir = self.task_data_dir

class BaselineConfig(object):
    def __init__(self):
        self.dialog_acts = ['sd', 'b', 'sv', '%', 'aa', 'ba', 'qy', 'ny', 'fc',
                'qw', 'nn', 'bk', 'fo_o_fw_"_by_bc', 'h', 'qy^d', 'bh', '^q', 
                'bf', 'na', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 
                'br', 'no', 'fp', 'qrr', 'arp_nd', 't3', 'oo_co_cc', 't1', 'bd',
                'aap_am', '^g', 'qw^d', 'fa', 'ft']
        self.joint_da_seg_recog_labels = ["I"] + ["E_"+da for da in self.dialog_acts]
        self.n_workers = 1
        
        # Management
        self.raw_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/swda"
        self.task_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/joint"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"

        # data splits
        self.train_dialog_list = f"{self.task_data_dir}/train.txt"
        self.test_dialog_list = f"{self.task_data_dir}/test.txt" 
        self.dev_dialog_list = f"{self.task_data_dir}/dev.txt" 

        # other
        self.ms_dir = "/g/ssli/data/treebank/ms_alignment/swb_ms98_transcriptions/"
        #self.out_dir = "/s0/ttmt001/swda_tsv"
        self.out_dir = self.task_data_dir

        # Pretrained embeddings (for initialization and evaluation)
        self.word_embedding_path = "/g/ssli/data/CTS-English/swbd_align/glove300.json"
        self.eval_word_embedding_path = "/g/ssli/data/CTS-English/swbd_align/glove300.json"

class TestConfig(object):
    def __init__(self):
        self.dialog_acts = ['sd', 'b', 'sv', '%', 'aa', 'ba', 'qy', 'ny', 'fc',
                'qw', 'nn', 'bk', 'fo_o_fw_"_by_bc', 'h', 'qy^d', 'bh', '^q', 
                'bf', 'na', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 
                'br', 'no', 'fp', 'qrr', 'arp_nd', 't3', 'oo_co_cc', 't1', 'bd',
                'aap_am', '^g', 'qw^d', 'fa', 'ft']
        self.joint_da_seg_recog_labels = ["I"] + ["E_"+da for da in self.dialog_acts]
        self.n_workers = 1
        
        # Management
        self.raw_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/swda"
        self.task_data_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/joint"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"

        # data splits
        self.test_dialog_list = f"{self.task_data_dir}/test.txt" 
        self.dev_dialog_list = f"{self.task_data_dir}/dev.txt" 

        # other
        self.ms_dir = "/g/ssli/data/treebank/ms_alignment/swb_ms98_transcriptions/"
        self.out_dir = self.task_data_dir

        # Pretrained embeddings (for initialization and evaluation)
        self.word_embedding_path = "/g/ssli/data/CTS-English/swbd_align/glove300.json"
        self.eval_word_embedding_path = "/g/ssli/data/CTS-English/swbd_align/glove300.json"

        # Path for predefined model:
        self.model_path = None


