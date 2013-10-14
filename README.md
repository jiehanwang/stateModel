stateModel
==========

Generate state models

This code is used for generating sign net model for each state. 
State is a sub-unit of sign in the system. Multiple states are learned to be one model.
Input: 
1. trajectory of P50,51,52,53,54.
2. hand shape feature: HOG. Or, the original key frames file in "2013****".
3. the clusted hand shape and the relationship for left, right and both. They are the output of code "cluster".
