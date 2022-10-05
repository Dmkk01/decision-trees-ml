# Machine Learning: Lab 1

Authors: Dominik Łasiński and David Glaas

## Assignment 0
The MONK-2 dataset should be the hardest dataset to learn because every variable 
is mostly independent of each other compared to the other datasets. This means that the entropy
is larger, furthermore it is hard to define the best question for collection of information gain and split
the set into subsets due to their independencies. 

The MONK-3 dataset should be the easiest to learn because it has more dependencies between the variables.
The MONK-1 dataset should have a difficulty that is between the MONK-2 and MONK-3.
## Assignment 1
```
monk_1 = d.entropy(m.monk1)
monk_2 = d.entropy(m.monk2)
monk_3 = d.entropy(m.monk3)
print(monk_1, monk_2, monk_3)
```

| Dataset | Entropy |
|---------|---------|
| MONK-1  |     1.0 |
| MONK-2  |      0.957117428264771   |
| MONK-3  |      0.9998061328047111   |

## Assignment 2
Entropy means unprictability.

Example of uniform distribution:
Low entropy:
Tossing coin probabilities: 0.5 => head and 0.5 => tail
So the entropy would be: −0.5·log2(0.5) − 0.5·log2(0.5) = 1

High entropy:
Normal dice probabilities: 1/6 => 1, 1/6 => 2, 1/6 => 3, 1/6 => 4, 1/6 => 5, 1/6 => 6
So the entropy would be = 6·(−1/6·log2(1/6)) = 2.58

If we take it up a notch and have a dice with 15 numbers there is even a lower chance of guessing which number
that will be displayed, thus its harder to predict and therefore the entropy is larger.

Example of non-uniform distribution:
Low entropy:
Weather in winter probabilities: 0.8 => below 15 degrees Celsius and 0.2 => over 15 degrees Celsius
So the entropy would be = -0.8·log2(0.8) - 0.2·log2(0.2) = 0.72

High entropy:
Fake dice probabilities: 0.1 => 1, 0.1 => 2, 0.1 => 3, 0.1 => 4, 0.1 => 5, 0.5 => 6
So the entropy would be = −5·0.1·log2(0.1) − 0.5·log2(0.5) = 2.16

So as a set becomes less uniformily distrobuted the entropy drops. Because its easier to predict the outcome.

## Assignment 3
```
monk_1_a_1 = d.averageGain(m.monk1, m.attributes[0])
monk_1_a_2 = d.averageGain(m.monk1, m.attributes[1])
monk_1_a_3 = d.averageGain(m.monk1, m.attributes[2])
monk_1_a_4 = d.averageGain(m.monk1, m.attributes[3])
monk_1_a_5 = d.averageGain(m.monk1, m.attributes[4])
monk_1_a_6 = d.averageGain(m.monk1, m.attributes[5])
print(monk_1_a_1, monk_1_a_2, monk_1_a_3, monk_1_a_4, monk_1_a_5, monk_1_a_6)

monk_2_a_1 = d.averageGain(m.monk2, m.attributes[0])
monk_2_a_2 = d.averageGain(m.monk2, m.attributes[1])
monk_2_a_3 = d.averageGain(m.monk2, m.attributes[2])
monk_2_a_4 = d.averageGain(m.monk2, m.attributes[3])
monk_2_a_5 = d.averageGain(m.monk2, m.attributes[4])
monk_2_a_6 = d.averageGain(m.monk2, m.attributes[5])
print(monk_2_a_1, monk_2_a_2, monk_2_a_3, monk_2_a_4, monk_2_a_5, monk_2_a_6)

monk_3_a_1 = d.averageGain(m.monk3, m.attributes[0])
monk_3_a_2 = d.averageGain(m.monk3, m.attributes[1])
monk_3_a_3 = d.averageGain(m.monk3, m.attributes[2])
monk_3_a_4 = d.averageGain(m.monk3, m.attributes[3])
monk_3_a_5 = d.averageGain(m.monk3, m.attributes[4])
monk_3_a_6 = d.averageGain(m.monk3, m.attributes[5])
print(monk_3_a_1, monk_3_a_2, monk_3_a_3, monk_3_a_4, monk_3_a_5, monk_3_a_6)
```

| Dataset | a_1 | a_2 | a_3 | a_4 | a_5 | a_6 |
|---------|-----|-----|-----|-----|-----|-----|
| MONK-1  |  0.07527255560831925   |  0.005838429962909286   |  0.00470756661729721   |  0.02631169650768228   |   0.28703074971578435  |  0.0007578557158638421   |
| MONK-2  |  0.0037561773775118823   |  0.0024584986660830532   |  0.0010561477158920196   |   0.015664247292643818  |   0.01727717693791797  |  0.006247622236881467   |
| MONK-3  |  0.007120868396071844   |   0.29373617350838865  |  0.0008311140445336207   |  0.002891817288654397   |  0.25591172461972755   |   0.007077026074097326  |

For MONK-1 - a_5 should be used because it has the most information gain

For MONK-2 - a_5 should be used because it has the most information gain

For MONK-3 - a_2 should be used because it has the most information gain

The definition of information gain could for example be the 
entropy of playing golf - (playing golf with respect to the weather)
or  playing golf - (playing golf with respect to the temperature) etc.

So in definition, it is the entropy of the outcome we are 
interested in - the different subsets.

We always choose the set with the highest information gain to be the root node.
## Assignment 4
When the information gain for a given subest S_k is maximised, 
its entropy becomes smaller. 
If we have more information received then it's much 
easier to make a decision based on the outcome.

We select an attibute to split in order to reduce the entropy in the subset. We do that so that we can create subsets that have lower entropy. 


So in conclusion, the information gained is maximized when the entropy of the subset s_k
becomes as small as possible in relation to the original dataset. Thus we choose the 
subset that provides the smallest uncertanty. We therefore now more about the subset and can
thus make a better prediction.

## Assignment 5
```
t_1 = d.buildTree(m.monk1, m.attributes);
print(d.check(t_1, m.monk1))
print(d.check(t_1, m.monk1test))

t_2 = d.buildTree(m.monk2, m.attributes);
print(d.check(t_2, m.monk2))
print(d.check(t_2, m.monk2test))

t_3 = d.buildTree(m.monk3, m.attributes);
print(d.check(t_3, m.monk3))
print(d.check(t_3, m.monk3test))
```

| Dataset | E_train | E_test |
|---------|---------|--------|
| MONK-1  |   1.0 - 1 = 0      |  1 - 0.8287037037037037 = 0.171     |
| MONK-2  |   1.0 - 1 = 0    |    1 - 0.6921296296296297 = 0.307   |
| MONK-3  |   1.0 - 1 = 0     |   1 - 0.9444444444444444 = 0.056    |

Our previous assumptions were correct. The MONK-2 was supposed to be the hardest dataset to learn due its parameters independency, thus creating the highest testing error.
Similarly, the MONK-3 was supposed to be the easiest dataset to learn due its parameters dependencies, thus creating the lowest testing error.

The training model is perfect and there is no error because we use the test set 
for learning and validation.
Thus there will be no error. The model will be perfect as long as we do this.

## Assignment 6

Pruning increases the bias and also reduces the variance.
Classification trees might have a high variance, so we use pruning to decrease it,
by removing non-critical subsets of the tree, thus reducing the complexity. 
We also reduce the entropy when pruning the tree. 
This means we decrease the possiblity of overfitting the tree, 
thus giving more realistic results. We can add complexity and reduce bias by increasing the
learning. 

Bias skews the result towards a certain outcome. 
Error due to Bias: The difference between the average (expected) prediction of
our model and the correct value.

Variance refers to the changes in a model when using different portions of the training set.
Error due to variance: the variability of a model prediction for a given data point 
between different realizations of a model.
## Assignment 7
<img src="/trees/task_7.png"/>
