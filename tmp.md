[E][72820][03-16][21:54:44:078][hb_dnn.cpp:292][example][DNN] [Task] input is null pointer
I20260316 21:54:44.078511 72820 infer_method.cc:131] hbDNNInferV2 failed, error code:-100001
F20260316 21:54:44.078531 72820 workflow_plugin.cc:248] Infer method failed
*** Check failure stack trace: ***
    @     0xaaaac464e054
    @     0xaaaac464df78
    @     0xaaaac464d8b0
    @     0xaaaac4650b7c
    @     0xaaaac462f89c
    @     0xffffb11a29cc
    @     0xffffb0ed0398
    @     0xffffb0f39e9c
latency.sh: line 15: 72778 Aborted                 (core dumped) ${app} --config_file=workflow_latency_image.json --log_level=1