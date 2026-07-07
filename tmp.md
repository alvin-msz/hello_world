ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_169035_1920244589_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_169237_2595365095_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168934_162031297_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_169037_281769784_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_169137_2232424241_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_169236_1002944753_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168936_3855341147_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168935_777696174_10>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168935_2674463211_11>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168934_2099506660_11>: No space left on device (28)
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168936_2506596224_11>: No space left on device (28)
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/functional.py:539: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:3637.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/dist-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [256, 256, 1, 1], strides() = [256, 1, 256, 256]
bucket_view.sizes() = [256, 256, 1, 1], strides() = [256, 1, 1, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Fatal Python error: Fatal Python error: Bus errorBus error

Fatal Python error: 

Thread 0xBus errorThread 0x

00007f4e8dffb640Fatal Python error: 00007f08f17fa640Thread 0x (most recent call first):
 (most recent call first):
Bus error00007e8a2595b640  File 

 (most recent call first):
  File "Thread 0x"/usr/lib/python3.10/socket.py  File 00007f354cfec640/usr/lib/python3.10/socket.py"" (most recent call first):
", line /usr/lib/python3.10/socket.py293  File , line " in "293accept, line /usr/lib/python3.10/socket.py in 
293  File "accept in ", line 
/usr/lib/python3.10/multiprocessing/connection.pyaccept293"  File 
 in , line "  File /usr/lib/python3.10/multiprocessing/connection.py609accept"" in 
, line /usr/lib/python3.10/multiprocessing/connection.pyaccept609  File "
 in , line "  File accept609/usr/lib/python3.10/multiprocessing/connection.py"
 in "/usr/lib/python3.10/multiprocessing/connection.py  File accept, line ""
609, line /usr/lib/python3.10/multiprocessing/connection.py  File " in 463", line accept in /usr/lib/python3.10/multiprocessing/connection.py463
accept" in , line   File 
accept463"  File /usr/lib/python3.10/multiprocessing/connection.py
 in ""  File accept/usr/lib/python3.10/multiprocessing/resource_sharer.py"
/usr/lib/python3.10/multiprocessing/resource_sharer.py, line "  File "463, line ", line  in 138/usr/lib/python3.10/multiprocessing/resource_sharer.py138accept in "_serve in 
, line 
  File _serve"138
  File  in /usr/lib/python3.10/multiprocessing/resource_sharer.py  File ""_serve"/usr/lib/python3.10/threading.py/usr/lib/python3.10/threading.py
, line "", line 138, line   File 953953 in " in  in run_serve/usr/lib/python3.10/threading.pyrun
"

  File   File , line   File ""953"/usr/lib/python3.10/threading.py/usr/lib/python3.10/threading.py/usr/lib/python3.10/threading.py" in "", line run, line , line 953
10161016 in   File  in  in _bootstrap_innerrun"_bootstrap_inner

/usr/lib/python3.10/threading.py
  File   File "  File """/usr/lib/python3.10/threading.py, line /usr/lib/python3.10/threading.py/usr/lib/python3.10/threading.py"1016"", line  in 973, line , line _bootstrap_inner in 1016973
_bootstrap in  in   File 
_bootstrap_bootstrap_inner"


/usr/lib/python3.10/threading.pyCurrent thread 0x
  File "00007f50c2ca6640Current thread 0x, line " (most recent call first):
00007f0b11e69640973/usr/lib/python3.10/threading.py  File  (most recent call first):
 in   File "", line _bootstrap"/usr/local/lib/python3.10/dist-packages/torch/storage.py
973/usr/local/lib/python3.10/dist-packages/torch/storage.py"
 in ", line Current thread 0x_bootstrap, line 52500007f0f6cbbf640
525 in  (most recent call first):

 in _share_fd_cpu_  File Current thread 0x_share_fd_cpu_
"00007f354d7ed640
  File /usr/local/lib/python3.10/dist-packages/torch/storage.py (most recent call first):
  File ""  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py, line ""/usr/local/lib/python3.10/dist-packages/torch/storage.py525, line /usr/local/lib/python3.10/dist-packages/torch/storage.py" in 450", line _share_fd_cpu_ in , line 450
wrapper525 in   File 
 in wrapper"  File _share_fd_cpu_
/usr/local/lib/python3.10/dist-packages/torch/storage.py
"  File "  File /usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py, line """450/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py/usr/local/lib/python3.10/dist-packages/torch/storage.py, line  in ""618, line wrapper, line  in 618
450reduce_storage in   File  in 
reduce_storagewrapper"  File 
"
/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py  File /usr/lib/python3.10/multiprocessing/reduction.py"  File "", line , line /usr/lib/python3.10/multiprocessing/reduction.py618"51" in , line  in /usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.pyreduce_storage"51dumps
 in   File , line 
dumps"618  File 
/usr/lib/python3.10/multiprocessing/reduction.py in "reduce_storage  File "/usr/lib/python3.10/multiprocessing/queues.py
", line "  File /usr/lib/python3.10/multiprocessing/queues.py51, line " in 244, line dumps in 244"
_feed in /usr/lib/python3.10/multiprocessing/reduction.py  File 
_feed""  File 
, line /usr/lib/python3.10/multiprocessing/queues.py"  File 51"/usr/lib/python3.10/threading.py" in , line "/usr/lib/python3.10/threading.pydumps244, line "
 in 953, line _feed in   File 953
run in   File "
run"
/usr/lib/python3.10/multiprocessing/queues.py  File /usr/lib/python3.10/threading.py  File """", line /usr/lib/python3.10/threading.py, line /usr/lib/python3.10/threading.py244"953" in , line  in , line _feed1016run1016
 in 
 in   File   File _bootstrap_inner_bootstrap_inner"

"/usr/lib/python3.10/threading.py  File   File /usr/lib/python3.10/threading.py"""", line /usr/lib/python3.10/threading.py/usr/lib/python3.10/threading.py, line 1016""953 in , line , line _bootstrap_inner in 973973
run in  in   File 
_bootstrap_bootstrap"  File 

/usr/lib/python3.10/threading.py"

"/usr/lib/python3.10/threading.pyThread 0xThread 0x, line "00007f91a5ead74000007fd756c97740973, line  (most recent call first):
 (most recent call first):
 in 1016  File   File _bootstrap" in "
/usr/lib/python3.10/selectors.py_bootstrap_inner/usr/lib/python3.10/selectors.py
"
"  File Thread 0x, line , line "00007f10bbbb4740416416/usr/lib/python3.10/threading.py (most recent call first):
 in  in "  File selectselect, line "

973/usr/lib/python3.10/selectors.py in   File   File "_bootstrap"", line /usr/lib/python3.10/multiprocessing/connection.py
/usr/lib/python3.10/multiprocessing/connection.py416"
" in , line Thread 0x, line select93100007fbbe223d740931
 in  (most recent call first):
 in   File waitwait  File "

"/usr/lib/python3.10/multiprocessing/connection.py  File   File /usr/lib/python3.10/selectors.py"""", line /usr/lib/python3.10/multiprocessing/connection.py/usr/lib/python3.10/multiprocessing/connection.py, line 931""416 in , line , line  in wait424424select
 in  in 
  File _poll_poll  File "

"/usr/lib/python3.10/multiprocessing/connection.py  File   File /usr/lib/python3.10/multiprocessing/connection.py"""", line /usr/lib/python3.10/multiprocessing/connection.py/usr/lib/python3.10/multiprocessing/connection.py, line 424931" in " in , line wait, line 
_poll257
257  File  in   File poll" in "
/usr/lib/python3.10/multiprocessing/connection.pypoll/usr/lib/python3.10/multiprocessing/connection.py  File "
"", line   File , line /usr/lib/python3.10/multiprocessing/queues.py257"424" in /usr/lib/python3.10/multiprocessing/queues.py in , line poll"_poll113

, line  in   File   File 113get in ""
get/usr/lib/python3.10/multiprocessing/queues.py/usr/lib/python3.10/multiprocessing/connection.py  File 
""/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py"  File , line "", line 113, line /usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py257 in 315" in get in poll, line 
_worker_loop315

 in   File   File _worker_loop  File ""
"/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py/usr/lib/python3.10/multiprocessing/queues.py/usr/lib/python3.10/multiprocessing/process.py"  File "", line , line ", line 113315 in /usr/lib/python3.10/multiprocessing/process.py108 in get" in _worker_loop
, line run
  File 108
  File  in "  File run"/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py"
/usr/lib/python3.10/multiprocessing/process.py"/usr/lib/python3.10/multiprocessing/process.py""  File , line ", line , line 315108/usr/lib/python3.10/multiprocessing/process.py314 in  in " in run, line _worker_loop_bootstrap
314

 in   File _bootstrap  File "
/usr/lib/python3.10/multiprocessing/process.py  File "  File ""/usr/lib/python3.10/multiprocessing/spawn.py", line /usr/lib/python3.10/multiprocessing/process.py"/usr/lib/python3.10/multiprocessing/spawn.py314"", line  in , line 129, line _bootstrap108 in 129
 in _main in run
_main  File 

  File "  File "/usr/lib/python3.10/multiprocessing/spawn.py  File "/usr/lib/python3.10/multiprocessing/spawn.py""/usr/lib/python3.10/multiprocessing/spawn.py", line "/usr/lib/python3.10/multiprocessing/process.py, line 129, line "116 in 116 in , line _main in spawn_main314
spawn_main
 in 
  File   File _bootstrap"  File "
/usr/lib/python3.10/multiprocessing/spawn.py"<string>"<string>", line "  File , line 116, line 1 in 1 in spawn_main in <module>
<module>

  File "<string>", line 1 in <module>
"/usr/lib/python3.10/multiprocessing/spawn.py", line 129 in _main
  File "/usr/lib/python3.10/multiprocessing/spawn.py", line 116 in spawn_main
  File "<string>", line 1 in <module>

Extension modules: numpy.core._multiarray_umath
Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.core._multiarray_tests, numpy.fft._pocketfft_internal, , numpy.linalg._umath_linalgnumpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.fft._pocketfft_internal, numpy.random._generator, numpy.random._common, torch._C, torch._C._dynamo.autograd_compiler, , numpy.random.bit_generatortorch._C._dynamo.eval_frame, , numpy.random._bounded_integerstorch._C._dynamo.guards, torch._C._dynamo.utils, numpy.random._mt19937, torch._C._fft, , numpy.random.mtrandtorch._C._linalg, , numpy.random._philoxtorch._C._nested, numpy.random._pcg64, torch._C._nn, torch._C._sparse, numpy.random._sfc64, torch._C._special
Extension modules: , numpy.core._multiarray_umathnumpy.random._generator, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, torch._C, numpy.fft._pocketfft_internal, torch._C._dynamo.autograd_compiler, , numpy.random._commontorch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, , numpy.random.bit_generatortorch._C._fft, numpy.random._bounded_integers, torch._C._linalg, numpy.random._mt19937, torch._C._nested, numpy.random.mtrand, , numpy.random._philoxtorch._C._nn, numpy.random._pcg64, torch._C._sparse, , numpy.random._sfc64torch._C._special, numpy.random._generator, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special
Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, PIL._imaging, numpy.linalg._umath_linalg, PIL._imagingft, av._core, av.logging, numpy.fft._pocketfft_internal, av.bytesource, av.buffer, av.audio.format, numpy.random._common, av.error, av.dictionary, av.container.pyio, av.option, av.descriptor, numpy.random.bit_generator, av.format, , av.utilsnumpy.random._bounded_integers, av.stream, , av.container.streamsnumpy.random._mt19937, av.sidedata.motionvectors, av.sidedata.sidedata, numpy.random.mtrand, av.opaque, , numpy.random._philoxav.packet, av.container.input, , numpy.random._pcg64, PIL._imagingav.container.output, , av.container.corenumpy.random._sfc64, av.codec.context, numpy.random._generator, av.video.format, PIL._imagingft, av.video.reformatter, av.plane, av._core, av.video.plane, , av.logging, torch._Cav.video.frame, av.bytesource, , PIL._imagingav.video.stream, av.buffer, , av.codec.hwaccel, torch._C._dynamo.autograd_compilerav.audio.format, , av.codec.codecPIL._imagingft, torch._C._dynamo.eval_frame, , , av._coreav.error, av.frame, torch._C._dynamo.guards, av.logging, av.dictionaryav.audio.layout, , torch._C._dynamo.utilsav.bytesource, , av.container.pyioav.audio.plane, av.buffer, , av.audio.frametorch._C._fft, , av.option, av.audio.formatav.audio.stream, , torch._C._linalg, av.descriptor, av.errorav.filter.link, , , , torch._C._nestedav.formatav.dictionaryav.filter.context, , , , av.container.pyiotorch._C._nnav.utilsav.filter.graph, av.option, , av.filter.filterav.stream, , av.descriptortorch._C._sparse, av.filter.loudnorm, , av.container.streams, av.format, torch._C._special, , av.audio.resamplerav.sidedata.motionvectorsav.utils, av.audio.codeccontext, , , av.sidedata.sidedataav.streamav.audio.fifo, av.opaque, , av.container.streams, av.bitstreamav.packet, av.sidedata.motionvectors, , , av.video.codeccontextav.container.inputav.sidedata.sidedata, , av.container.outputav.opaque, av.container.core, , av.packetav.codec.context, av.container.input, av.video.format, av.container.output, av.video.reformatter, av.container.core, av.plane, av.codec.context, , av.video.planeav.video.format, , av.video.frameav.video.reformatter, , av.video.streamav.plane, av.codec.hwaccel, av.video.plane, av.codec.codec, av.video.frame, , av.frameav.video.stream, , av.audio.layoutav.codec.hwaccel, , av.audio.planeav.codec.codec, , av.audio.frameav.frame, , av.audio.streamav.audio.layout, , av.filter.linkav.audio.plane, av.filter.context, av.audio.frame, av.filter.graph, av.audio.stream, av.filter.filter, av.filter.link, av.filter.loudnorm, av.filter.context, av.audio.resampler, av.filter.graph, , av.audio.codeccontextav.filter.filter, , av.audio.fifoav.filter.loudnorm, , av.bitstreamav.audio.resampler, av.video.codeccontext, av.audio.codeccontext, av.audio.fifo, av.bitstream, av.video.codeccontext, msgpack._cmsgpack, PIL._imaging, numba.core.typeconv._typeconv, numba._helperlib, numba._dynfunc, numba._dispatcher, PIL._imagingft, numba.core.runtime._nrt_python, numba.np.ufunc._internal, av._core, numba.experimental.jitclass._box, av.logging, scipy._lib._ccallback_c, , av.bytesourcescipy.sparse._sparsetools, scipy.sparse._csparsetools, av.buffer, scipy.sparse.linalg._isolve._iterative, scipy.linalg._fblas, , scipy.linalg._flapackav.audio.format, scipy.linalg._cythonized_array_utils, scipy.linalg._flinalg, scipy.linalg._solve_toeplitz, scipy.linalg._matfuncs_sqrtm_triu, av.error, scipy.linalg.cython_lapack, scipy.linalg.cython_blas, scipy.linalg._matfuncs_expm, scipy.linalg._decomp_update, av.dictionary, scipy.sparse.linalg._dsolve._superlu, scipy.sparse.linalg._eigen.arpack._arpack, , scipy.sparse.csgraph._toolsav.container.pyio, scipy.sparse.csgraph._shortest_path, scipy.sparse.csgraph._traversal, scipy.sparse.csgraph._min_spanning_tree, scipy.sparse.csgraph._flow, scipy.sparse.csgraph._matching, , av.optionscipy.sparse.csgraph._reordering, scipy.spatial._ckdtree, scipy._lib.messagestream, , av.descriptorscipy.spatial._qhull, scipy.spatial._voronoi, , av.formatscipy.spatial._distance_wrap, scipy.spatial._hausdorff, scipy.special._ufuncs_cxx, , scipy.special._ufuncsav.utils, scipy.special._specfun, scipy.special._comb, , scipy.special._ellip_harm_2av.stream, scipy.spatial.transform._rotation, av.container.streams, matplotlib._c_internal_utils, matplotlib._path, av.sidedata.motionvectors, kiwisolver._cext, av.sidedata.sidedata, matplotlib._image, av.opaque, av.packet, sklearn.__check_build._check_build, av.container.input, psutil._psutil_linux, av.container.output, , numpy.linalg.lapack_liteav.container.core, scipy.ndimage._nd_image, _ni_label, scipy.ndimage._ni_label, , scipy.optimize._minpack2av.codec.context, scipy.optimize._group_columns, scipy.optimize._trlib._trlib, av.video.format, scipy.optimize._lbfgsb, , msgpack._cmsgpack_moduleTNC, av.video.reformatter, scipy.optimize._moduleTNC, scipy.optimize._cobyla, scipy.optimize._slsqp, scipy.optimize._minpack, , scipy.optimize._lsq.givens_eliminationav.plane, scipy.optimize._zeros, scipy.optimize.__nnls, scipy.optimize._highs.cython.src._highs_wrapper, av.video.plane, scipy.optimize._highs._highs_wrapper, scipy.optimize._highs.cython.src._highs_constants, scipy.optimize._highs._highs_constants, scipy.linalg._interpolative, scipy.optimize._bglu_dense, , av.video.framescipy.optimize._lsap, scipy.optimize._direct, av.video.stream, scipy.integrate._odepack, , numba.core.typeconv._typeconvscipy.integrate._quadpack, av.codec.hwaccel, scipy.integrate._vode, scipy.integrate._dop, av.codec.codec, scipy.integrate._lsoda, , scipy.special.cython_specialav.frame, , numba._helperlibscipy.stats._stats, , beta_ufuncav.audio.layout, scipy.stats._boost.beta_ufunc, , binom_ufuncnumba._dynfunc, scipy.stats._boost.binom_ufunc, , av.audio.planenbinom_ufunc, scipy.stats._boost.nbinom_ufunc, hypergeom_ufunc, scipy.stats._boost.hypergeom_ufunc, av.audio.frame, ncf_ufunc, scipy.stats._boost.ncf_ufunc, scipy.interpolate._fitpack, , av.audio.streamnumba._dispatcher, scipy.interpolate.dfitpack, scipy.interpolate._bspl, scipy.interpolate._ppoly, av.filter.link, scipy.interpolate.interpnd, scipy.interpolate._rbfinterp_pythran, , av.filter.contextscipy.stats._biasedurn, scipy.stats._levy_stable.levyst, scipy._lib._uarray._uarray, av.filter.graph, scipy.stats._hypotests_pythran, av.filter.filter, scipy.stats._statlib, numba.core.runtime._nrt_python, scipy.stats._mvn, , scipy.stats._sobolav.filter.loudnorm, scipy.stats._qmc_cy, scipy.stats._unuran.unuran_wrapper, av.audio.resampler, pandas._libs.tslibs.dtypes, av.audio.codeccontext, pandas._libs.tslibs.base, pandas._libs.tslibs.np_datetime, av.audio.fifo, pandas._libs.tslibs.nattype, numba.np.ufunc._internal, pandas._libs.tslibs.timezones, pandas._libs.tslibs.ccalendar, , av.bitstreampandas._libs.tslibs.tzconversion, pandas._libs.tslibs.strptime, av.video.codeccontext, pandas._libs.tslibs.fields, pandas._libs.tslibs.timedeltas, pandas._libs.tslibs.timestamps, pandas._libs.properties, pandas._libs.tslibs.offsets, pandas._libs.tslibs.parsing, pandas._libs.tslibs.conversion, pandas._libs.tslibs.period, pandas._libs.tslibs.vectorized, pandas._libs.ops_dispatch, pandas._libs.missing, pandas._libs.hashtable, msgpack._cmsgpack, pandas._libs.algos, , pandas._libs.intervalnumba.experimental.jitclass._box, pandas._libs.tslib, pandas._libs.lib, pandas._libs.hashing, scipy._lib._ccallback_c, pyarrow.lib, pandas._libs.ops, pandas._libs.arrays, pandas._libs.index, pandas._libs.join, pandas._libs.sparse, pyarrow._compute, pandas._libs.reduction, scipy.sparse._sparsetools, , numba.core.typeconv._typeconvpandas._libs.indexing, , scipy.sparse._csparsetoolspandas._libs.internals, pandas._libs.writers, pandas._libs.window.aggregations, , numba._helperlibpandas._libs.window.indexers, scipy.sparse.linalg._isolve._iterative, pandas._libs.reshape, numba._dynfunc, pandas._libs.groupby, scipy.linalg._fblas, scipy.linalg._flapack, pandas._libs.testing, scipy.linalg._cythonized_array_utils, pandas._libs.parsers, , scipy.linalg._flinalg, pandas._libs.jsonnumba._dispatcher, scipy.linalg._solve_toeplitz, _cyutility, , sklearn._cyutilityscipy.linalg._matfuncs_sqrtm_triu, sklearn.utils._isfinite, , scipy.linalg.cython_lapacksklearn.utils.sparsefuncs_fast, scipy.linalg.cython_blas, sklearn.utils.murmurhash, scipy.linalg._matfuncs_expm, sklearn.utils._openmp_helpers, scipy.linalg._decomp_update, sklearn.metrics.cluster._expected_mutual_info_fast, numba.core.runtime._nrt_python, sklearn.preprocessing._csr_polynomial_expansion, scipy.sparse.linalg._dsolve._superlu, sklearn.preprocessing._target_encoder_fast, sklearn.metrics._dist_metrics, , scipy.sparse.linalg._eigen.arpack._arpacksklearn.metrics._pairwise_distances_reduction._datasets_pair, sklearn.utils._cython_blas, sklearn.metrics._pairwise_distances_reduction._base, sklearn.metrics._pairwise_distances_reduction._middle_term_computer, sklearn.utils._heap, , scipy.sparse.csgraph._toolssklearn.utils._sorting, sklearn.metrics._pairwise_distances_reduction._argkmin, , scipy.sparse.csgraph._shortest_pathsklearn.metrics._pairwise_distances_reduction._argkmin_classmode, , scipy.sparse.csgraph._traversalsklearn.utils._vector_sentinel, sklearn.metrics._pairwise_distances_reduction._radius_neighbors, , scipy.sparse.csgraph._min_spanning_treesklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode, sklearn.metrics._pairwise_fast, scipy.sparse.csgraph._flow, scipy.sparse.csgraph._matching, numba.np.ufunc._internal, , lmdb.cpythonscipy.sparse.csgraph._reordering, scipy.spatial._ckdtree, numba.experimental.jitclass._box, scipy._lib.messagestream, scipy.spatial._qhull, scipy._lib._ccallback_c, scipy.spatial._voronoi, scipy.spatial._distance_wrap, scipy.spatial._hausdorff, scipy.special._ufuncs_cxx, scipy.special._ufuncs, scipy.special._specfun, scipy.special._comb, , scipy.special._ellip_harm_2scipy.sparse._sparsetools, scipy.sparse._csparsetools (total: 242)
, scipy.spatial.transform._rotationERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
, scipy.sparse.linalg._isolve._iterative, scipy.linalg._fblas, scipy.linalg._flapack, matplotlib._c_internal_utils, scipy.linalg._cythonized_array_utils, scipy.linalg._flinalg, matplotlib._path, scipy.linalg._solve_toeplitz, scipy.linalg._matfuncs_sqrtm_triu, scipy.linalg.cython_lapack, scipy.linalg.cython_blas, scipy.linalg._matfuncs_expm, scipy.linalg._decomp_update, kiwisolver._cext, scipy.sparse.linalg._dsolve._superlu, scipy.sparse.linalg._eigen.arpack._arpack, scipy.sparse.csgraph._tools, scipy.sparse.csgraph._shortest_path, scipy.sparse.csgraph._traversal, matplotlib._image, scipy.sparse.csgraph._min_spanning_tree, scipy.sparse.csgraph._flow, scipy.sparse.csgraph._matching, scipy.sparse.csgraph._reordering, scipy.spatial._ckdtree, scipy._lib.messagestream, scipy.spatial._qhull, scipy.spatial._voronoi, scipy.spatial._distance_wrap, sklearn.__check_build._check_build, scipy.spatial._hausdorff, scipy.special._ufuncs_cxx, scipy.special._ufuncs, scipy.special._specfun, scipy.special._comb, , psutil._psutil_linuxscipy.special._ellip_harm_2, scipy.spatial.transform._rotation, matplotlib._c_internal_utils, matplotlib._path, numpy.linalg.lapack_lite, scipy.ndimage._nd_image, _ni_label, scipy.ndimage._ni_label, kiwisolver._cext, scipy.optimize._minpack2, scipy.optimize._group_columns, scipy.optimize._trlib._trlib, matplotlib._image, scipy.optimize._lbfgsb, _moduleTNC, scipy.optimize._moduleTNC, scipy.optimize._cobyla, scipy.optimize._slsqp, scipy.optimize._minpack, scipy.optimize._lsq.givens_elimination, scipy.optimize._zeros, scipy.optimize.__nnls, , sklearn.__check_build._check_buildscipy.optimize._highs.cython.src._highs_wrapper, scipy.optimize._highs._highs_wrapper, scipy.optimize._highs.cython.src._highs_constants, scipy.optimize._highs._highs_constants, scipy.linalg._interpolative, scipy.optimize._bglu_dense, , psutil._psutil_linuxscipy.optimize._lsap, scipy.optimize._direct, scipy.integrate._odepack, scipy.integrate._quadpack, scipy.integrate._vode, scipy.integrate._dop, scipy.integrate._lsoda, scipy.special.cython_special, scipy.stats._stats, numpy.linalg.lapack_lite, beta_ufunc, scipy.stats._boost.beta_ufunc, scipy.ndimage._nd_image, binom_ufunc, scipy.stats._boost.binom_ufunc, _ni_label, nbinom_ufunc, scipy.ndimage._ni_label, scipy.stats._boost.nbinom_ufunc, hypergeom_ufunc, scipy.optimize._minpack2, scipy.stats._boost.hypergeom_ufunc, scipy.optimize._group_columns, ncf_ufunc, scipy.stats._boost.ncf_ufunc, scipy.interpolate._fitpack, scipy.optimize._trlib._trlib, scipy.interpolate.dfitpack, scipy.interpolate._bspl, scipy.interpolate._ppoly, scipy.optimize._lbfgsb, , scipy.interpolate.interpnd_moduleTNC, scipy.optimize._moduleTNC, , scipy.interpolate._rbfinterp_pythranscipy.optimize._cobyla, scipy.optimize._slsqp, , scipy.stats._biasedurnscipy.optimize._minpack, scipy.stats._levy_stable.levyst, scipy.optimize._lsq.givens_elimination, scipy.optimize._zeros, scipy._lib._uarray._uarray, scipy.optimize.__nnls, scipy.optimize._highs.cython.src._highs_wrapper, scipy.optimize._highs._highs_wrapper, scipy.stats._hypotests_pythran, scipy.optimize._highs.cython.src._highs_constants, scipy.optimize._highs._highs_constants, scipy.linalg._interpolative, scipy.stats._statlib, scipy.optimize._bglu_dense, scipy.optimize._lsap, scipy.stats._mvn, scipy.stats._sobol, scipy.optimize._direct, scipy.stats._qmc_cy, scipy.integrate._odepack, , scipy.integrate._quadpackscipy.stats._unuran.unuran_wrapper, scipy.integrate._vode, scipy.integrate._dop, scipy.integrate._lsoda, pandas._libs.tslibs.dtypes, pandas._libs.tslibs.base, scipy.special.cython_special, scipy.stats._stats, pandas._libs.tslibs.np_datetime, pandas._libs.tslibs.nattype, beta_ufunc, scipy.stats._boost.beta_ufunc, pandas._libs.tslibs.timezones, binom_ufunc, , pandas._libs.tslibs.ccalendarscipy.stats._boost.binom_ufunc, nbinom_ufunc, pandas._libs.tslibs.tzconversion, scipy.stats._boost.nbinom_ufunc, hypergeom_ufunc, scipy.stats._boost.hypergeom_ufunc, pandas._libs.tslibs.strptime, ncf_ufunc, scipy.stats._boost.ncf_ufunc, pandas._libs.tslibs.fields, scipy.interpolate._fitpack, pandas._libs.tslibs.timedeltas, , scipy.interpolate.dfitpackpandas._libs.tslibs.timestamps, , pandas._libs.propertiesscipy.interpolate._bspl, , scipy.interpolate._ppolypandas._libs.tslibs.offsets, , pandas._libs.tslibs.parsingscipy.interpolate.interpnd, pandas._libs.tslibs.conversion, scipy.interpolate._rbfinterp_pythran, pandas._libs.tslibs.period, pandas._libs.tslibs.vectorized, scipy.stats._biasedurn, pandas._libs.ops_dispatch, scipy.stats._levy_stable.levyst, pandas._libs.missing, pandas._libs.hashtable, scipy._lib._uarray._uarray, pandas._libs.algos, pandas._libs.interval, scipy.stats._hypotests_pythran, pandas._libs.tslib, pandas._libs.lib, scipy.stats._statlib, pandas._libs.hashing, scipy.stats._mvn, scipy.stats._sobol, scipy.stats._qmc_cy, scipy.stats._unuran.unuran_wrapper, pandas._libs.tslibs.dtypes, pandas._libs.tslibs.base, pyarrow.lib, pandas._libs.tslibs.np_datetime, pandas._libs.tslibs.nattype, pandas._libs.tslibs.timezones, pandas._libs.tslibs.ccalendar, pandas._libs.tslibs.tzconversion, pandas._libs.ops, pandas._libs.tslibs.strptime, pandas._libs.tslibs.fields, pandas._libs.tslibs.timedeltas, pandas._libs.tslibs.timestamps, pandas._libs.arrays, pandas._libs.properties, pandas._libs.tslibs.offsets, pandas._libs.tslibs.parsing, pandas._libs.tslibs.conversion, pandas._libs.tslibs.period, pandas._libs.tslibs.vectorized, pandas._libs.ops_dispatch, pandas._libs.index, pandas._libs.join, pandas._libs.sparse, pyarrow._compute, pandas._libs.missing, pandas._libs.hashtable, pandas._libs.reduction, pandas._libs.algos, pandas._libs.indexing, pandas._libs.interval, pandas._libs.tslib, pandas._libs.lib, pandas._libs.hashing, pandas._libs.internals, pandas._libs.writers, pyarrow.lib, pandas._libs.window.aggregations, pandas._libs.ops, pandas._libs.window.indexers, pandas._libs.reshape, pandas._libs.arrays, pandas._libs.groupby, pandas._libs.index, pandas._libs.join, pandas._libs.sparse, pandas._libs.testing, pyarrow._compute, pandas._libs.parsers, pandas._libs.reduction, pandas._libs.indexing, pandas._libs.json, pandas._libs.internals, pandas._libs.writers, pandas._libs.window.aggregations, pandas._libs.window.indexers, _cyutility, sklearn._cyutility, sklearn.utils._isfinite, pandas._libs.reshape, sklearn.utils.sparsefuncs_fast, sklearn.utils.murmurhash, pandas._libs.groupby, sklearn.utils._openmp_helpers, sklearn.metrics.cluster._expected_mutual_info_fast, sklearn.preprocessing._csr_polynomial_expansion, pandas._libs.testing, sklearn.preprocessing._target_encoder_fast, , sklearn.metrics._dist_metricspandas._libs.parsers, sklearn.metrics._pairwise_distances_reduction._datasets_pair, sklearn.utils._cython_blas, pandas._libs.json, sklearn.metrics._pairwise_distances_reduction._base, sklearn.metrics._pairwise_distances_reduction._middle_term_computer, sklearn.utils._heap, sklearn.utils._sorting, sklearn.metrics._pairwise_distances_reduction._argkmin, sklearn.metrics._pairwise_distances_reduction._argkmin_classmode, sklearn.utils._vector_sentinel, sklearn.metrics._pairwise_distances_reduction._radius_neighbors, _cyutility, sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode, sklearn._cyutility, sklearn.utils._isfinite, sklearn.metrics._pairwise_fast, sklearn.utils.sparsefuncs_fast, sklearn.utils.murmurhash, sklearn.utils._openmp_helpers, sklearn.metrics.cluster._expected_mutual_info_fast, lmdb.cpython, sklearn.preprocessing._csr_polynomial_expansion, sklearn.preprocessing._target_encoder_fast, sklearn.metrics._dist_metrics, sklearn.metrics._pairwise_distances_reduction._datasets_pair, sklearn.utils._cython_blas, sklearn.metrics._pairwise_distances_reduction._base, sklearn.metrics._pairwise_distances_reduction._middle_term_computer, sklearn.utils._heap, sklearn.utils._sorting, sklearn.metrics._pairwise_distances_reduction._argkmin, sklearn.metrics._pairwise_distances_reduction._argkmin_classmode, sklearn.utils._vector_sentinel, sklearn.metrics._pairwise_distances_reduction._radius_neighbors, sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode, sklearn.metrics._pairwise_fast, lmdb.cpython (total: 242)
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
 (total: 242)
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
, msgpack._cmsgpack, numba.core.typeconv._typeconv, numba._helperlib, numba._dynfunc, numba._dispatcher, numba.core.runtime._nrt_python, numba.np.ufunc._internal, numba.experimental.jitclass._box, scipy._lib._ccallback_c, scipy.sparse._sparsetools, scipy.sparse._csparsetools, scipy.sparse.linalg._isolve._iterative, scipy.linalg._fblas, scipy.linalg._flapack, scipy.linalg._cythonized_array_utils, scipy.linalg._flinalg, scipy.linalg._solve_toeplitz, scipy.linalg._matfuncs_sqrtm_triu, scipy.linalg.cython_lapack, scipy.linalg.cython_blas, scipy.linalg._matfuncs_expm, scipy.linalg._decomp_update, scipy.sparse.linalg._dsolve._superlu, scipy.sparse.linalg._eigen.arpack._arpack, scipy.sparse.csgraph._tools, scipy.sparse.csgraph._shortest_path, scipy.sparse.csgraph._traversal, scipy.sparse.csgraph._min_spanning_tree, scipy.sparse.csgraph._flow, scipy.sparse.csgraph._matching, scipy.sparse.csgraph._reordering, scipy.spatial._ckdtree, scipy._lib.messagestream, scipy.spatial._qhull, scipy.spatial._voronoi, scipy.spatial._distance_wrap, scipy.spatial._hausdorff, scipy.special._ufuncs_cxx, scipy.special._ufuncs, scipy.special._specfun, scipy.special._comb, scipy.special._ellip_harm_2, scipy.spatial.transform._rotation, matplotlib._c_internal_utils, matplotlib._path, kiwisolver._cext, matplotlib._image, sklearn.__check_build._check_build, psutil._psutil_linux, numpy.linalg.lapack_lite, scipy.ndimage._nd_image, _ni_label, scipy.ndimage._ni_label, scipy.optimize._minpack2, scipy.optimize._group_columns, scipy.optimize._trlib._trlib, scipy.optimize._lbfgsb, _moduleTNC, scipy.optimize._moduleTNC, scipy.optimize._cobyla, scipy.optimize._slsqp, scipy.optimize._minpack, scipy.optimize._lsq.givens_elimination, scipy.optimize._zeros, scipy.optimize.__nnls, scipy.optimize._highs.cython.src._highs_wrapper, scipy.optimize._highs._highs_wrapper, scipy.optimize._highs.cython.src._highs_constants, scipy.optimize._highs._highs_constants, scipy.linalg._interpolative, scipy.optimize._bglu_dense, scipy.optimize._lsap, scipy.optimize._direct, scipy.integrate._odepack, scipy.integrate._quadpack, scipy.integrate._vode, scipy.integrate._dop, scipy.integrate._lsoda, scipy.special.cython_special, scipy.stats._stats, beta_ufunc, scipy.stats._boost.beta_ufunc, binom_ufunc, scipy.stats._boost.binom_ufunc, nbinom_ufunc, scipy.stats._boost.nbinom_ufunc, hypergeom_ufunc, scipy.stats._boost.hypergeom_ufunc, ncf_ufunc, scipy.stats._boost.ncf_ufunc, scipy.interpolate._fitpack, scipy.interpolate.dfitpack, scipy.interpolate._bspl, scipy.interpolate._ppoly, scipy.interpolate.interpnd, scipy.interpolate._rbfinterp_pythran, scipy.stats._biasedurn, scipy.stats._levy_stable.levyst, scipy._lib._uarray._uarray, scipy.stats._hypotests_pythran, scipy.stats._statlib, scipy.stats._mvn, scipy.stats._sobol, scipy.stats._qmc_cy, scipy.stats._unuran.unuran_wrapper, pandas._libs.tslibs.dtypes, pandas._libs.tslibs.base, pandas._libs.tslibs.np_datetime, pandas._libs.tslibs.nattype, pandas._libs.tslibs.timezones, pandas._libs.tslibs.ccalendar, pandas._libs.tslibs.tzconversion, pandas._libs.tslibs.strptime, pandas._libs.tslibs.fields, pandas._libs.tslibs.timedeltas, pandas._libs.tslibs.timestamps, pandas._libs.properties, pandas._libs.tslibs.offsets, pandas._libs.tslibs.parsing, pandas._libs.tslibs.conversion, pandas._libs.tslibs.period, pandas._libs.tslibs.vectorized, pandas._libs.ops_dispatch, pandas._libs.missing, pandas._libs.hashtable, pandas._libs.algos, pandas._libs.interval, pandas._libs.tslib, pandas._libs.lib, pandas._libs.hashing, pyarrow.lib, pandas._libs.ops, pandas._libs.arrays, pandas._libs.index, pandas._libs.join, pandas._libs.sparse, pyarrow._compute, pandas._libs.reduction, pandas._libs.indexing, pandas._libs.internals, pandas._libs.writers, pandas._libs.window.aggregations, pandas._libs.window.indexers, pandas._libs.reshape, pandas._libs.groupby, pandas._libs.testing, pandas._libs.parsers, pandas._libs.json, _cyutility, sklearn._cyutility, sklearn.utils._isfinite, sklearn.utils.sparsefuncs_fast, sklearn.utils.murmurhash, sklearn.utils._openmp_helpers, sklearn.metrics.cluster._expected_mutual_info_fast, sklearn.preprocessing._csr_polynomial_expansion, sklearn.preprocessing._target_encoder_fast, sklearn.metrics._dist_metrics, sklearn.metrics._pairwise_distances_reduction._datasets_pair, sklearn.utils._cython_blas, sklearn.metrics._pairwise_distances_reduction._base, sklearn.metrics._pairwise_distances_reduction._middle_term_computer, sklearn.utils._heap, sklearn.utils._sorting, sklearn.metrics._pairwise_distances_reduction._argkmin, sklearn.metrics._pairwise_distances_reduction._argkmin_classmode, sklearn.utils._vector_sentinel, sklearn.metrics._pairwise_distances_reduction._radius_neighbors, sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode, sklearn.metrics._pairwise_fast, lmdb.cpython (total: 242)
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/usr/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/reductions.py", line 618, in reduce_storage
    fd, size = storage._share_fd_cpu_()
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 450, in wrapper
    return fn(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/storage.py", line 525, in _share_fd_cpu_
    return super()._share_fd_cpu_(*args, **kwargs)
RuntimeError: unable to write to file </torch_168937_2346743167_30>: No space left on device (28)
2026-07-07 08:56:04,312 ERROR [ddp_trainer.py:463] Node[3] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1251, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 424, in _poll
    r = wait([self], timeout)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/usr/lib/python3.10/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 169036) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 542, in fit
    _, batch = next(self.data_loader_pr)
  File "/usr/local/lib/python3.10/dist-packages/hat/profiler/profilers.py", line 263, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/generator.py", line 20, in prefetch_iterator
    for val in it:
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1458, in _next_data
    idx, data = self._get_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1420, in _get_data
    success, data = self._try_get_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1264, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 169036) exited unexpectedly

2026-07-07 08:56:04,319 ERROR [ddp_trainer.py:463] Node[2] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1251, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 424, in _poll
    r = wait([self], timeout)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/usr/lib/python3.10/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 169136) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 542, in fit
    _, batch = next(self.data_loader_pr)
  File "/usr/local/lib/python3.10/dist-packages/hat/profiler/profilers.py", line 263, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/generator.py", line 20, in prefetch_iterator
    for val in it:
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1458, in _next_data
    idx, data = self._get_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1420, in _get_data
    success, data = self._try_get_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1264, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 169136) exited unexpectedly

2026-07-07 08:56:20,596 ERROR [ddp_trainer.py:463] Node[1] Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1251, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/usr/lib/python3.10/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 424, in _poll
    r = wait([self], timeout)
  File "/usr/lib/python3.10/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/usr/lib/python3.10/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 169236) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 457, in _with_exception
    fn(*args)
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 186, in train_entrance
    trainer.fit()
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/loop_base.py", line 542, in fit
    _, batch = next(self.data_loader_pr)
  File "/usr/local/lib/python3.10/dist-packages/hat/profiler/profilers.py", line 263, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.10/dist-packages/hat/utils/generator.py", line 20, in prefetch_iterator
    for val in it:
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 708, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1458, in _next_data
    idx, data = self._get_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1420, in _get_data
    success, data = self._try_get_data()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1264, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 169236) exited unexpectedly

W0707 08:56:21.963000 168571 torch/multiprocessing/spawn.py:169] Terminating process 168712 via signal SIGTERM
W0707 08:56:21.965000 168571 torch/multiprocessing/spawn.py:169] Terminating process 168713 via signal SIGTERM
W0707 08:56:21.965000 168571 torch/multiprocessing/spawn.py:169] Terminating process 168715 via signal SIGTERM
ERROR:__main__:train failed! process 2 terminated with exit code 1
Traceback (most recent call last):
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 287, in <module>
    raise e
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 273, in <module>
    train(
  File "/open_explorer/samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py", line 254, in train
    launch(
  File "/usr/local/lib/python3.10/dist-packages/hat/engine/ddp_trainer.py", line 426, in launch
    mp.spawn(
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/spawn.py", line 340, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/spawn.py", line 296, in start_processes
    while not context.join():
  File "/usr/local/lib/python3.10/dist-packages/torch/multiprocessing/spawn.py", line 204, in join
    raise ProcessExitedException(
torch.multiprocessing.spawn.ProcessExitedException: process 2 terminated with exit code 1