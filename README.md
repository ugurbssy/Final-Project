# Final-Project
Market Basket Analysis

Association rules analysis is a technique to uncover how items are associated to each other. There are three common ways to measure association.
Measure 1: Support. This says how popular an itemset is, it is number of times appear in total number of transaction. in other word we say frequency of item.
Measure 2: Confidence. This says how likely item Y is purchased when item X is purchased, expressed as {X -> Y}. This is measured by the proportion of transactions with item X, in which item Y also appears.
Measure 3: Lift. it is ratio of expected confidance to observed confidance. it is described as confidance of Y when item X was already known(x/y) to the confidance of Y when X item is unknown. in other words confidance of Y w.r.t. x and confiadnce of Y without X (means both are independent to each other).
support = occurance of item / total no of transaction.
confidance = support ( X Union Y) / support(X).
lift = support (X Union Y)/ support(X) * support(Y) 
