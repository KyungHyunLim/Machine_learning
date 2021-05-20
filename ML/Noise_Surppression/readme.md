[Experiment environment] <br>

{Hardware} <br>
PC: Galaxy flex NT950 <br>
CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz   1.50 GHz <br>
GPU: Nvidia GeForce MX250 <br>
RAM: 16.0GB <br> <br>

{Software} <br>
OS: Window    10 <br>
Paython       3.7 <br>
Pytorch       1.7.1 <br>
CUDA          10.2 <br>
Nvidia driver 442.23 <br>

[Sources]<br>
{16KHz_DownSampling} <br>
48kHz -> 16kHz <br>
<br>
{Preprocessing} <br>
Slicing and Windowing <br>
overlapping 50% <br>
list -> torch tensor <br>
<br>
{Model} <br>
Define 'Discriminator' and 'Noisecanceler(Generator)' <br>

{Model - Structure} <br>
Noise Discriminator ans Clean Discriminator <br>
x : clean speech <br>
x~: Noisy speech <br>
x^: Enhanced speech <br>
n : x~ - x <br>
n^: x~ - x^ <br>
<img src='https://user-images.githubusercontent.com/72729802/118740899-f4ace780-b887-11eb-80cd-22b6651078e4.png'/> <br>
Whole Architecture <br>
<img src='https://user-images.githubusercontent.com/72729802/118740902-f5457e00-b887-11eb-84c9-007544a15db1.png'/> <br>

{Train} <br>
- Order -
0. Generator using MSE
1. clean discriminator
2. noise discriminator
3. Generator
