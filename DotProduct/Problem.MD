# Dot Product

Implement a GPU program that computes the dot product of two vectors containing 32-bit floating point numbers. The dot 
product is the sum of the products of the corresponding elements of two vectors.

## Initial version
    -> all threads compute the local block level vector dot product and update the global sum 
    -> it is done using the atomics and we have only one shared variable so basically all threads got serialized 
    -> to remove the we have to something similler to reduction so we have some parallelization

## improved version 