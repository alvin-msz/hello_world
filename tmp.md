python3 samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py --nuscenes-root ./data/nuscenes --output-dir ./samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/mini_data/flashocc_henet/ --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/custom/flashocc-r50-M0_bevfusionocc_horizon_2.py --version v1.0-trainval --split val  
`aidisdk` dependency is not available.
[warn] config 缺少 march，使用默认 March.NASH_M
2026-06-25 09:13:33,024 WARNING [/usr/local/lib/python3.10/dist-packages/horizon_plugin_pytorch/fx/fx_helper.py:230] wrap usage has been changed, please pass necessary args
/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/maptr/sparse_head.py:157: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/maptr/sparse_decoder.py:121: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
/usr/local/lib/python3.10/dist-packages/hat/models/task_modules/maptr/sparse_decoder.py:139: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @autocast(enabled=False)
[warn] cfg.model 构建失败: BN has not registered in any of registry ['HAT_OBJECT_REGISTRY'] and is not a class, which is not allowed. 

Traceback (most recent call last):
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 681, in <module>
    main()
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 608, in main
    export_model, _ = init_export_model(args.config, DEFAULT_TOOLS_DIR, cfg_ns=cfg)
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 515, in init_export_model
    raise RuntimeError(
RuntimeError: 无法从配置构建带 export_reference_points 的模型，请确认 --config 含 calibration/qat predictor 及有效 checkpoint