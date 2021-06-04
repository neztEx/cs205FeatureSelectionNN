# cs205FeatureSelectionNN
Given a dataset the program trains our model to return a set of features that give us the greatest accuracy of classification.

# How to Run
```python3 main.py```


## Demo 

```
select data set: 
 1.)small
 2.)large
1
Do you want to normalize: 
 1.)yes
 2.)no
1
Please select your Search: 
1) Forward Selection 
2) Back Elimination
1
      0         1         2         3         4         5         6         7         8         9         10
0    2.0 -0.461063  0.549678 -1.189446  1.169312  0.571171  0.893501  1.686738  0.044216 -0.145083 -0.396306
1    2.0 -0.601840  0.289135 -1.844733 -1.465964  1.592010  1.485387  0.332755 -0.222883 -1.153897  0.204305
2    2.0  1.283188 -1.902748 -0.397598  2.200506  1.366532 -0.107691 -0.237627  2.099086 -0.097889  0.569751
3    2.0 -0.080278  0.376642 -0.560772  0.238434  0.469418  1.692627 -0.987300 -1.593965  1.054153  1.548179
4    2.0 -0.881066  1.344910 -0.414045  0.403070 -1.740356 -0.876321 -0.275150  0.360838  0.217322 -1.015905
..   ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
295  2.0 -0.322680  0.776619  0.587143  1.073797 -0.360731 -0.870816  0.795103 -0.636621 -1.288353 -0.764396
296  2.0  0.147232  0.644972 -1.151710 -0.489451  1.437249 -0.757371 -0.392235 -1.067596 -0.237481 -0.241346
297  1.0 -0.175749  0.412691 -0.506668  0.776715  0.291144  0.633935  1.014956 -0.228542 -0.414770 -0.079493
298  2.0  0.306514 -0.760983 -0.169911 -0.461604  0.630558 -1.207516  0.589838 -0.026435  0.757696 -1.344398
299  2.0 -0.850626  1.157697 -0.315755 -0.834533 -1.187486  0.207092 -0.469999  2.059605 -0.059458 -0.448571

[300 rows x 11 columns]
11
On the 1th level
 -Current Features [] Considering adding the 1 feature - %0.6966666666666667
 -Current Features [] Considering adding the 2 feature - %0.6733333333333333
 -Current Features [] Considering adding the 3 feature - %0.6933333333333334
 -Current Features [] Considering adding the 4 feature - %0.7133333333333334
 -Current Features [] Considering adding the 5 feature - %0.68
 -Current Features [] Considering adding the 6 feature - %0.8633333333333333
 -Current Features [] Considering adding the 7 feature - %0.69
 -Current Features [] Considering adding the 8 feature - %0.6866666666666666
 -Current Features [] Considering adding the 9 feature - %0.6566666666666666
 -Current Features [] Considering adding the 10 feature - %0.6733333333333333
On level 1, I added feature 6 to current set with accuracy %0.8633333333333333
...
```