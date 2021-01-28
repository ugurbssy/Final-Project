Market Basket Analysis

By analysing the buying behaviour of customers, try to find out which are the products that are bought together by the customers.


```python
pip install apyori
```

    Requirement already satisfied: apyori in c:\users\asus\anaconda3\lib\site-packages (1.1.2)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from apyori import apriori
```


```python
import sys,getopt
import requests
import csv
import Orange
from Orange.data import Table,Domain, DiscreteVariable, ContinuousVariable
from orangecontrib.associate.fpgrowth import *
from scipy import sparse
import scipy.stats as ss

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from PIL import Image

```


```python
df = pd.read_csv(r"C:\Users\ASUS\Downloads\market.csv", header=None)
df.head()
# View the dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(0,inplace=True)
df.head()
#replaced the null values wiht zero to see nicer
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Here, the data represents the items a customer bought at a particular time at grocery market. 
#Dataset contains 20 columns and 7501 rows where each row represents the names of items in each respective columns.
```


```python
#[7501 rows x 20 columns]
```

We need a data in form of 'list' for using Aprori Algorithm.


```python
transactions = [] 
for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])                    
```


```python
print(type(transactions))
```

    <class 'list'>
    


```python
transactions[0]
```




    ['shrimp',
     'almonds',
     'avocado',
     'vegetables mix',
     'green grapes',
     'whole weat flour',
     'yams',
     'cottage cheese',
     'energy drink',
     'tomato juice',
     'low fat yogurt',
     'green tea',
     'honey',
     'salad',
     'mineral water',
     'salmon',
     'antioxydant juice',
     'frozen smoothie',
     'spinach',
     'olive oil']



The goal here is to apply Apriori algorithm on the dataset.
Rules of support, confidence and lift as described as follows; 
1)Support: It is calculated to check how much popular a given item is. 
It is measured by the proportion of transactions in which an itemset appears. 
For example, there are 100 people who bought something from market, 
amoung those 100 people, there are 20 people who bought "X" product. 
Therefore, the support of people who bought "X" product will be (20/100 = 20%).
2)Confidence: It is calculated to check how likely if item "X" is purchased when item Y is purchased. 
This is measured by the proportion of transactions with item "X", in which item Y also appears. 
3)Lift: It is calculated to measure how likely item Y is purchased when item "X" is purchased, 
while controlling for how popular item Y is. 

Apriori Algorithm
We have provided min_support, min_confidence, min_lift of sample-set and mi_length to find the rule.
Min_support  = 5(5 times a day) * 7 (7 days a week) 35/ 7501 = 0.0045 (dataset is for one week period)
Min_confidence = set it lower to get more relations between products,so selected confidence level is 0.25
Min_lift = In order to get some relevant rules,setting min_lift to 3.
Min_length = 2 (at least two products is taken in the rule)


```python
from apyori import apriori

rules = apriori(transactions,min_support=0.0045,min_confidence=0.25,min_lift=3,min_length=2)
results = list(rules)

# we convert the rules into a list because it is easier to see the results in list format.
```


```python
df_results = pd.DataFrame(results)

```


```python
print("There are {} Relation derived.".format(len(results)))
```

    There are 18 Relation derived.
    


```python
#df_results.head()
df_results.sort_values(by=['support'],ascending=False)
#Sorted according to support value descending
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>items</th>
      <th>support</th>
      <th>ordered_statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>(ground beef, herb &amp; pepper)</td>
      <td>0.015998</td>
      <td>[((herb &amp; pepper), (ground beef), 0.3234501347...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(ground beef, spaghetti, frozen vegetables)</td>
      <td>0.008666</td>
      <td>[((spaghetti, frozen vegetables), (ground beef...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(whole wheat pasta, olive oil)</td>
      <td>0.007999</td>
      <td>[((whole wheat pasta), (olive oil), 0.27149321...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(mineral water, frozen vegetables, shrimp)</td>
      <td>0.007199</td>
      <td>[((mineral water, shrimp), (frozen vegetables)...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(spaghetti, tomatoes, frozen vegetables)</td>
      <td>0.006666</td>
      <td>[((spaghetti, tomatoes), (frozen vegetables), ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(ground beef, mineral water, herb &amp; pepper)</td>
      <td>0.006666</td>
      <td>[((mineral water, herb &amp; pepper), (ground beef...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(ground beef, spaghetti, herb &amp; pepper)</td>
      <td>0.006399</td>
      <td>[((spaghetti, herb &amp; pepper), (ground beef), 0...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(ground beef, spaghetti, shrimp)</td>
      <td>0.005999</td>
      <td>[((ground beef, shrimp), (spaghetti), 0.523255...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(pasta, escalope)</td>
      <td>0.005866</td>
      <td>[((pasta), (escalope), 0.3728813559322034, 4.7...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(mushroom cream sauce, escalope)</td>
      <td>0.005733</td>
      <td>[((mushroom cream sauce), (escalope), 0.300699...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(ground beef, tomato sauce)</td>
      <td>0.005333</td>
      <td>[((tomato sauce), (ground beef), 0.37735849056...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(chocolate, frozen vegetables, shrimp)</td>
      <td>0.005333</td>
      <td>[((chocolate, shrimp), (frozen vegetables), 0....</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(ground beef, spaghetti, grated cheese)</td>
      <td>0.005333</td>
      <td>[((spaghetti, grated cheese), (ground beef), 0...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(pasta, shrimp)</td>
      <td>0.005066</td>
      <td>[((pasta), (shrimp), 0.3220338983050847, 4.506...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(ground beef, cooking oil, spaghetti)</td>
      <td>0.004799</td>
      <td>[((ground beef, cooking oil), (spaghetti), 0.5...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(milk, frozen vegetables, olive oil)</td>
      <td>0.004799</td>
      <td>[((frozen vegetables, olive oil), (milk), 0.42...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>(light cream, chicken)</td>
      <td>0.004533</td>
      <td>[((light cream), (chicken), 0.2905982905982905...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(milk, mineral water, spaghetti, frozen vegeta...</td>
      <td>0.004533</td>
      <td>[((milk, mineral water, spaghetti), (frozen ve...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(results[2])
#Print any item to see our rule
```

    RelationRecord(items=frozenset({'pasta', 'escalope'}), support=0.005865884548726837, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'escalope'}), confidence=0.3728813559322034, lift=4.700811850163794)])
    

Accodring to result above, we can say that pasta and escalope are bought frequently. 
The support for pasta is 0.0058. The confidence for this rule is 0.37288. means that out of all 
the transactions containing pasta, 37.28% of the transactions are likely to contain escalope as well. 
Lastly, lift of 4.70 shows that the escalope is 4.70 more likely to be bought by the customers that buy pasta, 
compared to its default sale. 


```python
for item in results:
    # first index of the inner list
    
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " => " + items[1])

    # second index of the inner list which gives support value
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("************************************")
```

    Rule: light cream => chicken
    Support: 0.004532728969470737
    Confidence: 0.29059829059829057
    Lift: 4.84395061728395
    ************************************
    Rule: mushroom cream sauce => escalope
    Support: 0.005732568990801226
    Confidence: 0.3006993006993007
    Lift: 3.790832696715049
    ************************************
    Rule: pasta => escalope
    Support: 0.005865884548726837
    Confidence: 0.3728813559322034
    Lift: 4.700811850163794
    ************************************
    Rule: ground beef => herb & pepper
    Support: 0.015997866951073192
    Confidence: 0.3234501347708895
    Lift: 3.2919938411349285
    ************************************
    Rule: ground beef => tomato sauce
    Support: 0.005332622317024397
    Confidence: 0.3773584905660377
    Lift: 3.840659481324083
    ************************************
    Rule: whole wheat pasta => olive oil
    Support: 0.007998933475536596
    Confidence: 0.2714932126696833
    Lift: 4.122410097642296
    ************************************
    Rule: pasta => shrimp
    Support: 0.005065991201173177
    Confidence: 0.3220338983050847
    Lift: 4.506672147735896
    ************************************
    Rule: chocolate => frozen vegetables
    Support: 0.005332622317024397
    Confidence: 0.29629629629629634
    Lift: 3.1084175084175087
    ************************************
    Rule: ground beef => cooking oil
    Support: 0.004799360085321957
    Confidence: 0.5714285714285714
    Lift: 3.2819951870487856
    ************************************
    Rule: ground beef => spaghetti
    Support: 0.008665511265164644
    Confidence: 0.31100478468899523
    Lift: 3.165328208890303
    ************************************
    Rule: milk => frozen vegetables
    Support: 0.004799360085321957
    Confidence: 0.4235294117647058
    Lift: 3.2684095860566447
    ************************************
    Rule: mineral water => frozen vegetables
    Support: 0.007199040127982935
    Confidence: 0.30508474576271183
    Lift: 3.200616332819722
    ************************************
    Rule: spaghetti => tomatoes
    Support: 0.006665777896280496
    Confidence: 0.3184713375796179
    Lift: 3.341053850607991
    ************************************
    Rule: ground beef => spaghetti
    Support: 0.005332622317024397
    Confidence: 0.3225806451612903
    Lift: 3.283144395325426
    ************************************
    Rule: ground beef => mineral water
    Support: 0.006665777896280496
    Confidence: 0.39062500000000006
    Lift: 3.975682666214383
    ************************************
    Rule: ground beef => spaghetti
    Support: 0.006399146780429276
    Confidence: 0.3934426229508197
    Lift: 4.004359721511667
    ************************************
    Rule: ground beef => spaghetti
    Support: 0.005999200106652446
    Confidence: 0.5232558139534884
    Lift: 3.005315360233627
    ************************************
    Rule: milk => mineral water
    Support: 0.004532728969470737
    Confidence: 0.28813559322033894
    Lift: 3.0228043143297376
    ************************************
    
