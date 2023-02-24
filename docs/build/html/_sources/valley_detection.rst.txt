Finding Valleys 
=====

To detect valleys use the following functions. 

An example to detect and validate racket valleys

>>> vallyes = findValleys(x)
>>> vallyes = groupValleys(vallyes)
>>> #only for racket
>>> vallyes = checkValleysSanity(vallyes, wall_valleys) 


.. autofunction:: Utils.Valleys.findValleys


.. autofunction:: Utils.Valleys.groupValleys


.. autofunction:: Utils.Valleys.checkValleysSanity
