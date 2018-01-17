# Gaussian Processes on the GPU

This tool provides a simple implementation of Gaussian processes for
Nvidia GPUs in CUDA C, for testing and evaluation purposes.  A
benchmark, comparing a multithreaded CPU implementation using OpenMP
with the one for GPUs is provided.

## Installation

The following packages are required as external dependencies:

 - Armadillo (and LAPACK): http://arma.sourceforge.net
 - NLopt: http://ab-initio.mit.edu/wiki/index.php/NLopt
 - libconfig: http://www.hyperrealm.com/libconfig/
 - tvmet http://tvmet.sourceforge.net/
 
On Ubuntu, the following packages can be installed using apt (there is
no package for tvmet).

```
libarmadillo-dev
libnlopt0
libconfig++-dev
```

For tvmet:
```
$ wget https://downloads.sourceforge.net/project/tvmet/Tar.Gz_Bz2%20Archive/1.7.2/tvmet-1.7.2.tar.bz2
$ tar xf tvmet-1.7.2.tar.bz2
$ cd tvmet-1.7.2
$ ./configure
$ make
$ sudo make install
```

## Description

The training data for the Gaussian process are internal energies
returned by a particular analytic form of equation of state (which is
implemented as `Romenskii::e_internal` in `bench/romenskii.hpp`,
although the sampling and evaluation was done elsewhere).  These are
`values-2500.dat`.

It has a hard-coded squared exponential covariance kernel, with eight
hyperparameters specified in a way suitable for the problem.

The benchmark measures time for predictions from the Gaussian process
(not training).  A number of predictions can be made at a time in a
single batch, to reduce the overhead of kernel launches.

## Usage

Produce the Gaussian process to be used for the benchmark:

```
$ make gp_train
$ ./gp_train <values-2500.dat
```

which outputs `eos-example.gp`.

The configuration for `gp_train` is read from `gp_train.cfg`, in
libconfig format.  Change the value of `N` in this file to use a
different number of values (the data file does not need to be modified
in this case).

Compile and run the benchmark:

```
$ make bench/benchmark
$ OMP_NUM_THREADS=T ./bench/benchmark N
```

where `T` should be replaced by the number of OpenMP threads to use,
and `N` is the number of simultaneous predictions to make.


## Results

The following results were produced in 2015 on then-current hardware, and
could currently do with updating.  Double precision was used
throughout.

- The CPU was an Intel Xeon E5-2650 v2 with 16 cores at 2.60 GHz, with ideal double precision performance of 333 GFLOPS.
- The GPU was an Nvidia K20, with ideal double precision performance of 1170 GFLOPS.

<table>
  <tr>
    <th rowspan="2">Training set size</th>
    <th rowspan="2">Number of evaluations (batch size)</th>
    <th colspan="3">Time per evaluation (&mu;s)</th>
  </tr>
  <tr>
    <th>CPU, 1 thread</th>
    <th>CPU, 16 threads</th>
    <th>GPU</th>
  </tr>
  <tr> <td rowspan="6">100</td>
       <td>1</td>      <td>95.1</td> <td>5570</td> <td>1000</td> </tr>
  <tr> <td>10</td>     <td>21.9</td> <td>456</td>  <td>100</td>  </tr>
  <tr> <td>100</td>    <td>14.5</td> <td>50.5</td> <td>10.2</td> </tr>
  <tr> <td>1000</td>   <td>13.7</td> <td>8.01</td> <td>1.26</td> </tr>
  <tr> <td>10000</td>  <td>13.6</td> <td>2.32</td> <td>0.360</td></tr>
  <tr> <td>20000</td>  <td>13.6</td> <td>1.53</td> <td>0.320</td></tr>
  <tr> <td rowspan="6">2000</td>
       <td>1</td>      <td>371</td> <td>4690</td> <td>1080</td>  </tr>
  <tr> <td>10</td>     <td>284</td> <td>626</td>  <td>108</td>   </tr>
  <tr> <td>100</td>    <td>274</td> <td>93.2</td> <td>14.3</td>  </tr>
  <tr> <td>1000</td>   <td>273</td> <td>23.1</td> <td>4.92</td>  </tr>
  <tr> <td>10000</td>  <td>274</td> <td>19.1</td> <td>4.04</td>  </tr>
  <tr> <td>20000</td>  <td>274</td> <td>18.4</td> <td>3.99</td>  </tr>
</table>

There is a constant-time overhead associated with OpenMP in the
multithreaded CPU implementation, and a similar overhead in the GPU
implementation for the launch of the kernel (which includes some
transfer of data to and from the device).  The overheads can be seen
to become less significant for the larger runs (in both training set
size and number of evaluations in a batch).
