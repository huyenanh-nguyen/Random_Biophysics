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

    Args:
        events (list): list of events
    
    Returns:
        List: a list with the sum of each events of a joint distribution
    
    """

    return [sum(events[i]) for i in range(len(events))]

px = marginal_probability_distribution(x_events)
py = marginal_probability_distribution(y_events)

print("px(i) = ", px)
print("py(i) = ", py)


# [conditional probability distribution]______________________________________________________________