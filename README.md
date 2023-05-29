# Quadruped Drake

This repository contains code for the simulation and control of quadruped robots using [Drake](https://drake.mit.edu).

![](demo.gif)

## Dependencies

- [Drake](https://drake.mit.edu), compiled with Python bindings
- [LCM](https://lcm-proj.github.io/)
- [Eigen](http://eigen.tuxfamily.org/)
- [Ipopt](https://projects.coin-or.org/Ipopt)
- [ifopt](https://github.com/ethz-adrl/ifopt)
- [CMake](https://cmake.org/cmake/help/v3.0/)
- [Numpy](https://numpy.org)

## Installation

Clone this repository: `git clone https://github.com/vincekurtz/quadruped_drake.git`

Compile C\+\+ code (includes TOWR and custom LCM bindings for interface with drake):
```
mkdir -p build
cd build
cmake ..
make
```

## Usage

Start the drake visualizer `bazel-bin/tools/drake_visualizer`.

Run the simulation script `./simulate.py`.

## 关于环境配置

从源代码安装Drake

1.从源代码安装，参考
```angular2html
git clone https://github.com/RobotLocomotion/drake.git
```
2.运行环境配置
```angular2html
sudo ./setup/ubuntu/install_prereqs.sh
```
该指令会自动检查环境并安装LCM，Eigen,Ipopt等

3.安装Bazel(自行google)

4.用bazel编译drake
注意bazel编译需要指定cpu数，不然线程数太多电脑会卡死
```angular2html
cd /path/to/drake
bazel build --jobs=10//...                 # Build the entire project.
bazel test //...                  # Build and test the entire project.

bazel build --config=clang //...  # Build using Clang on Ubuntu.
bazel test --config=clang //...   # Build and test using Clang on Ubuntu.
```
5.安装ifopt(参考上文)

6.编译towr
本项目使用的towr是更改版本的，在原来的基础上加入了minicheetah模型和LCM通信
```angular2html
mkdir -p build
cd build
cmake ..
make
```
7.运行drake_visualizer
```angular2html
cd /path/to/drake
bazel-bin/tools/drake_visualizer
```
8.运行仿真程序
```angular2html
./simulate.py
```

9.运行跨越障碍程序
在simulate.py开头部分取消注释，设置
```angular2html
### TODO:运行跨越障碍程序
planning_method = "towr"
control_method = "MPTC"
sim_time = 3.5
```
运行即可