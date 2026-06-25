python3 samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py --nuscenes-root ./data/nuscenes --output-dir ./samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/mini_data/flashocc_henet/ --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/custom/flashocc_henet_lss_occ3d_nuscenes_bevfusion.py --version v1.0-trainval --split val  
`aidisdk` dependency is not available.
Traceback (most recent call last):
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 755, in <module>
    main()
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 678, in main
    cfg = _load_py_config(args.config)
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 88, in _load_py_config
    return _dict_to_namespace(cfg_dict)
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in _dict_to_namespace
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in <dictcomp>
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in _dict_to_namespace
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in <dictcomp>
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in _dict_to_namespace
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in <dictcomp>
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in _dict_to_namespace
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in <dictcomp>
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 71, in _dict_to_namespace
    ns = types.SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
TypeError: keywords must be strings