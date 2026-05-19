../../aarch64/bin/example --config_file=workflow_latency.json --log_level=1
I00000000 00:00:00.000000 41325 vlog_is_on.cc:195] RAW: Set VLOG level for "*" to 1
I20260316 19:07:18.547969 41325 simple_example.cc:42] EXAMPLE_SYSTEM
I20260316 19:07:18.548019 41325 simple_example.cc:43] EXAMPLE_REPORT
[BPU][[BPU_MONITOR]][281473399455776][INFO]BPULib verison(2, 2, 15)[f21ee84]!
[DNN]: 3.13.6_(4.7.5 HBRT)
I20260316 19:07:19.969326 41325 raw_data_iterator.cc:220] loop_able: 0, time_diff_ms: -1
I20260316 19:07:20.042387 41366 tensor_utils.cc:459] Quanti only support int8_t, int16_t, BOOL8, int64.
[ERROR][][mem_log.c:104] [12338.55590][41325:41366][MEM_ALLOCATOR] <hb_mem_flush_buf_with_vaddr:3647> Invalid NULL virtual address.
[E][41366][03-16][19:07:20:042][hb_ucp_sys.cpp:155][example][UCP] Clean cached memory failed, size: 0
[E][41367][03-16][19:07:20:042][dnn_task.cpp:325][example][DNN] [Task] Model flashocc-r50-M0_bevfusionocc_horizon_2 input index 2 's sys mem size is not enough, required: 36, given: 0
[E][41367][03-16][19:07:20:042][dnn_task.cpp:256][example][DNN] [Task] Model flashocc-r50-M0_bevfusionocc_horizon_2 validate input[2] failed!
[E][41367][03-16][19:07:20:042][dnn_task.cpp:181][example][DNN] [Task] invalid input
I20260316 19:07:20.042783 41367 infer_method.cc:131] hbDNNInferV2 failed, error code:-100001
F20260316 19:07:20.042805 41367 workflow_plugin.cc:248] Infer method failed
*** Check failure stack trace: ***
    @     0xaaaac2f78144
    @     0xaaaac2f78068
    @     0xaaaac2f779a0
    @     0xaaaac2f7ac6c
    @     0xaaaac2f5998c
    @     0xffffb06b29cc
    @     0xffffb03e0398
    @     0xffffb0449e9c
latency.sh: line 16: 41325 Aborted                 (core dumped) ${app} --config_file=workflow_latency.json --log_level=1