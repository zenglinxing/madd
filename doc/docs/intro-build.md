Build & Install
===

Configure & Build
---

```bash
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<your install path> -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest
cmake --install .
```

If your generator is Visual Studio, you should use `cmake --build . --config Release`, `ctest -C Release` and `cmake --install . --config Release` instead.

Build Options & Environments
---

Apart from common options `-DCMAKE_BUILD_TYPE=Release` and `-DCMAKE_INSTALL_PREFIX=`, there are multiple options & variables to set.

| options | Default | Description | 
| :------ | :-----: | :---------- |
| `-DENABLE_BENCHMARK` | OFF | Build benchmark code. Requires benchmark library by Google |
| `-DENABLE_CUDA` | OFF | Build CUDA functions |
| `-DCMAKE_CUDA_ARCHITECTURES` | 86 | CUDA architecures code. This variable should refer to<br>https://developer.nvidia.com/cuda-gpus and your NVIDIA GPU model |
|`-DENABLE_TEST` | ON | Build tests. There are too many tests. |
| `-DENABLE_MULTITHREAD` | OFF | Enable multithread. Requires thread_base from my git repository |
| `-DENABLE_QUADPRECISION` | OFF | 128-bit float support from quadmath |
<!-- | `-DMADD_THREAD_API` | C++ | If your OS is Windows, suggest to switch to Windows<br>If your OS is Linux/Unix/MacOS, suggest to switch to pthread | -->