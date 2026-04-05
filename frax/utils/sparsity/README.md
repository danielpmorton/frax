Note: These sparse operations are interesting and theoretically have lower FLOPs than just calling a dense inversion

But, there's a lot of overhead that XLA can't fuse together perfectly, and for mass matrices on the order of what is typical for a humanoid, it's more efficient to just call a dense method (like `jsp.linalg.inv(a, assume_a="pos")`)