1. JIT compilation takes forever: When launching models, all 4 workers are trying to JIT-compile the same extension simultaneously, which creates a race condition and filesystem lock contention on Perlmutter's shared filesystem (Lustre).


