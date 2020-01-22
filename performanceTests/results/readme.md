## Perfomance tests for mallocMC
- Tests for parallel allocation of memory on gpu
- as reference the default 'new' operator can be used.


# Usage of perfomance_test1
- three params; input via location:
  perfomance_test1 useMallocMC chunkSize maxAllocatedCunks

# Device memory kernels and their usage

- each kernel is in control of an array of pointers with given global length
- on host for each run the pointer where to (de-)allocate memory is globally generated and used as a parameter of the kernel calls
- a allocation kernel requests memory with a byte size chunkSize*allocSize; where allocSize is a pseudorandom number between 1 and maxAllocatedCunks


# Evaluation of performance metrics

- start with an warm up phase: pseudorandom memory (de-) allocation multiple times
- testing: measure (in kernel itself) the clockCount for memory allocation multiple times
- Reduce clockCounts over all threads -> write minimum, average and maximum to file

# Visualization
- using visualize.py
