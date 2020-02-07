# CUDA
cuda tutorial

How to solve the problem "nvcc fatal : Path to libdevice library not specified
"

I had the same error msg:nvcc fatal : Path to libdevice library not specified

I tried this from nvidia support
$ export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}

and it was completely resolved


**************************************************************************************
How to solve "nvprof XXX" error
firstly make a softlink of nvprof(usually in /usr/local/CUDA/bin) in /usr/local/bin, or add "nvprof" path in environmental variable PATH.
secondly using "sudo nvprof" instead of "nvprof"
