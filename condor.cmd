executable	  	= job_debug.sh
universe		= vanilla
copy_to_spool   	= False
getenv		  	= True
nice_user		= True
request_memory 		= 8000
request_cpus		= 1
request_gpus   		= 1
+GPUJob		 	= "true"
notification		= complete
should_transfer_files 	= NO
requirements		= NikolaHost == "fred"
name            = baseline
log			= /homes/ttmt001/transitory/dialog-act-prediction/joint_seg_da/condor_logs/log.$(Cluster).$(Process)
error			= /homes/ttmt001/transitory/dialog-act-prediction/joint_seg_da/condor_logs/err.$(name).$(Cluster)
output			= /homes/ttmt001/transitory/dialog-act-prediction/joint_seg_da/condor_logs/$(name)_log.txt
queue
