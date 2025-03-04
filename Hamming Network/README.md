A Python implementation of the Hamming Network and MaxNet for classification, using a TB diagnosis example.
It will:

- Classify a patient's symptoms using Hamming Network (find the closest match).
- Refine the result using MaxNet (ensure only the best match remains active).

The output is like this:

  _Hamming Network Output (Before MaxNet): [ 8. 10.]_
  
  _MaxNet Output (After Suppression): [0.    4.688]_

  _Final Classification: TB-Infected_



- The Hamming Network initially assigns scores (8 for normal, 10 for TB-infected).
- The MaxNet Network suppresses the weaker response, leaving only the strongest.
- The final classification is TB-Infected.
