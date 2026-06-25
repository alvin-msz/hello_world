Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/config.py", line 230, in __getattr__
    return getattr(self._cfg_dict, name)
AttributeError: 'dict' object has no attribute 'march'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/config.py", line 234, in __getattr__
    return self.__getitem__(name)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/config.py", line 240, in __getitem__
    return self._cfg_dict.__getitem__(name)
KeyError: 'march'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 663, in <module>
    main()
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 590, in main
    export_model, _ = init_export_model(args.config, DEFAULT_TOOLS_DIR)
  File "/open_explorer/samples/ucp_tutorial/dnn/ai_benchmark/runtime/qat/script/custom/flashocc_henet/prepare_nuscenes_mini_inputs.py", line 414, in init_export_model
    horizon.march.set_march(cfg.march)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/config.py", line 236, in __getattr__
    raise AttributeError(name)
AttributeError: march
