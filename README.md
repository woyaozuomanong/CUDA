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

**************************************************************************************
ubuntu升级内核后会造成原有的nvidia驱动不可用，此时可以先在启动时选择进入旧的内核，把新的内核删掉，卸载旧的nvidia驱动，重新安装nvidia驱动即可。
安装时注意gcc和g++的版本要在7.0以上，即/usr/bin/gcc和/usr/bin/g++的版本。如果之前降级过这两个软链接的版本，要先升回来，安装好驱动后再降回去。
