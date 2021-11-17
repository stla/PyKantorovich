
Welcome to PyKantorovich’s documentation!
*****************************************


Indices and tables
******************

*  `Index <genindex.rst>`_

*  `Module Index <py-modindex.rst>`_

*  `Search Page <search.rst>`_

**pykantorovich.extreme_joinings(mu, nu, number_type='fraction',
prettyprint=True)**

   Extreme joinings of two probabiity measures.

   :Parameters:
      *  **mu** (*array-like*) – A probability vector.

      *  **nu** (*array-like*) – A probability vector. Must have the
         same length as *mu*.

      *  **number_type** (*str*) – The type to use for the
         calculations, either *“fraction”* or *“float”*.

      *  **prettyprint** (*bool*) – Whether to pretty-print the
         results (especially when *number_type=”fraction”*).

   :Returns:
      The extreme joinings of *mu* and *nu*.

   :Return type:
      list

   -[ Examples ]-

   >>> mu = ['1/2','1/4','1/4']
   >>> nu = ['1/4','1/4','1/2']
   >>> joinings = extreme_joinings(mu, nu)
   [['1/4' '0' '1/4']
    ['0' '1/4' '0']
    ['0' '0' '1/4']]
   [['1/4' '0' '1/4']
    ['0' '0' '1/4']
    ['0' '1/4' '0']]
   [['1/4' '1/4' '0']
    ['0' '0' '1/4']
    ['0' '0' '1/4']]
   [['0' '1/4' '1/4']
    ['1/4' '0' '0']
    ['0' '0' '1/4']]
   [['0' '1/4' '1/4']
    ['0' '0' '1/4']
    ['1/4' '0' '0']]
   [['0' '0' '1/2']
    ['0' '1/4' '0']
    ['1/4' '0' '0']]
   [['0' '0' '1/2']
    ['1/4' '0' '0']
    ['0' '1/4' '0']]

**pykantorovich.kantorovich(mu, nu, distance_matrix='0-1',
method='cdd', number_type='fraction', prettyprint=True)**

   Kantorovich distance between two probabiity measures on a finite
   set.

   :Parameters:
      *  **mu** (*array-like*) – Array-like representing a
         probability.

      *  **nu** (*array-like*) – Array-like representing a
         probability. Must have the same length as *mu*.

      *  **distance_matrix** (*n x n matrix*) – The cost matrix. The
         default is “0-1”, the zero-or-one distance. Note that the
         implemented calculation of the Kantorovich distance is
         totally useless in this case, because it equals *1.0 -
         np.sum(np.minimum(mu, nu))*.

      *  **method** (*string*) – The method to use. Can be “cdd”,
         “sparse”, or “cvx”.

      *  **number_type** (*string*) – For method=”cdd” only. Can be
         “float” or “fraction”. The default is “fraction”.

      *  **prettyprint** (*bool*) – This is only for *method=”cdd”*
         and *number_type=”fraction”*. This prints a more readable
         result.

   :Returns:
   
         The Kantorovich distance between *mu* and *nu* and a joining
         of *mu* and *nu* for which the Kantorovich distance is the probability
         that the two margins differ.

   -[ Examples ]-

   >>> mu = ['1/7','2/7','4/7']
   >>> nu = ['1/4','1/4','1/2']
   >>> kantorovich(mu, nu)
   {
    distance: 3/28
    joining:
    [['1/7' '0' '0']
    ['1/28' '1/4' '0']
    ['1/14' '0' '1/2']]
    optimal: yes
   }
