import numpy as np

"""
Conditional and mutual information.
I kinda give my calculator to my niece... it is kinda hideous to to the math with my phone.

Let us consider the following probability table for the joint distribution of random variables X
and Y:

        Row Y = 1 : 1/8, 1/16, 1/32, 1/32
        Row Y = 2 : 1/16, 1/8, 1/32, 1/32
        Row Y = 3 : 1/16, 1/16, 1/16, 1/16
        Row y = 4 : 1/4, 0, 0, 0

Please derive from this table (in the unit bit) ∀i,j ∈{1,...,4}:
- marignal probability distribution p(x) and p(y).
- conditional probability distribution p(j|i) and p(i|j)
- And with the aid of the latter H(X), H(Y), H(X|Y), H(Y|X), H(X,Y) and I(X;Y)

"""

# [given values]___________________________________________________________________

joint_distribution = np.array(
    [[1/8, 1/16, 1/32, 1/32],
    [1/16, 1/8, 1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4, 0, 0, 0]]
)

x_events = [joint_distribution[...,i] for i in range(joint_distribution.shape[0])]
y_events = [joint_distribution[j] for j in range(joint_distribution.shape[0])]


# [marginal probability distribution]_______________________________________________


def marginal_probability_distribution(events : list):
    """
    Given a known joint distribution of two discrete random variables, say, X and Y, the marginal distribution of either variable  (X for example)  is the probability distribution of X when the values of Y are not taken into consideration. 
    This can be calculated by summing the joint probability distribution over all values of Y. 
    Naturally, the converse is also true: the marginal distribution can be obtained for Y by summing over the separate values of X.

    $$ p_{X}(x_{i})=\sum _{j}p(x_{i},y_{j}),\quad {\text{and}}\quad p_{Y}(y_{j})=\sum _{i}p(x_{i},y_{j}) $$
    
    Let us consider the following probability table for the joint distribution of random variables X
    and Y:

            Row Y = 1 : 1/8, 1/16, 1/32, 1/32
            Row Y = 2 : 1/16, 1/8, 1/32, 1/32
            Row Y = 3 : 1/16, 1/16, 1/16, 1/16
            Row y = 4 : 1/4, 0, 0, 0
    
    for the x-events would be the values in the columns, hence first x-events px(1) and analog for py(1) would be:
        
        ```
        px(1) = 1/8, 1/16, 1/16, 1/4
        py(1) = 1/8, 1/16, 1/32, 1/32
        ```


    Args:
        events (array_like): list of events
    
    Returns:
        List: a list with the sum of each events of a joint distribution
    
    """

    return [sum(events[i]) for i in range(len(events))]

px = marginal_probability_distribution(x_events)
py = marginal_probability_distribution(y_events)

print("px(i) = ", px)
print("py(i) = ", py)


# [conditional probability distribution]_____________________________________________

def conditional_probability_distribution(events: list, marginal_events: list):
    """
    Given two jointly distributed random variables X and Y, the conditional probability distribution of Y given X is the probability distribution of Y when X is known to be a particular value; in some cases the conditional probabilities may be expressed as functions containing the unspecified value x of X as a parameter. 
    When both X and Y are categorical variables, a conditional probability table is typically used to represent the conditional probability. The conditional distribution contrasts with the marginal distribution of a random variable, which is its distribution without reference to the value of the other variable.

    Let us consider the following probability table for the joint distribution of random variables X
    and Y:

        Row Y = 1 : 1/8, 1/16, 1/32, 1/32
        Row Y = 2 : 1/16, 1/8, 1/32, 1/32
        Row Y = 3 : 1/16, 1/16, 1/16, 1/16
        Row y = 4 : 1/4, 0, 0, 0
    
    for the x-events would be the values in the columns, hence first x-events px(1) and analog for py(1) would be:
        
        ```
        px(1) = 1/8, 1/16, 1/16, 1/4
        py(1) = 1/8, 1/16, 1/32, 1/32
        ```

    Args:
        events (array_like): list of events
        marginal_events (array_like): calculated marginalevents for X or Y

    Returns:
        List: a list with the conditional probability distribution of i base on j (analog for j on i)
    """
    
    p_ij = []
    for i in range(len(events)):
        quote = [float(j/marginal_events[i]) for _,j in enumerate(events[i])]
        p_ij.append(quote)
    
    return np.array(p_ij)

p_ij = conditional_probability_distribution(x_events, px)
p_ji = conditional_probability_distribution(y_events, py)

print("p(i|j) = ", p_ij)
print("p(j|i) = ", p_ji)


# [H(X), H(Y), H(X|Y), H(Y|X), H(X,Y) and I(X; Y)]__________________________________________________________

def shannon_entropy(probability):
    """
    The gain in information when measuring x and y equals the gain of information
    when measuring only x plus the gain of information when measuring y under
    the condition that x is known.
    x causes y to a certain extent, if the events are not completely independent.
    It also follows from the previous that
    H(x)+ H(y) ≥ H(x,y) = H(x) + H(y|x)

    Shannon Entropy is in bits (log base 2)

    Args:
        probability (array_like): array of the probability table for the joint distribution of random variables X and Y:

    Returns:
        float: return shannon entropy in bits
    """

    joint = -np.sum(probability * np.log2(probability, where=(probability > 0)))
    return joint



print("H(X,Y) = ", shannon_entropy(joint_distribution)) 
print("H(X) = ",shannon_entropy(np.array(px)))
print("H(Y) = ",shannon_entropy(np.array(py)))
print("H(X|Y) = ", shannon_entropy(joint_distribution) - shannon_entropy(np.array(py)))
print("H(Y|X) = ", shannon_entropy(joint_distribution) - shannon_entropy(np.array(px)))
print("I(X;Y) = ", shannon_entropy(np.array(px)) - (shannon_entropy(joint_distribution) - shannon_entropy(np.array(py))))