# Prison-Break_Machine_Learning

**Objective:** To solve a prison-break problem using machine learning algorithm

Here is a prisoner, Steve, He is known as a fugitive from prison.

there is a [prison](https://github.com/rabieifk/Prison-Break_Machine_Learning/blob/master/Prison.png) with spetial room.


At any stage, Steve can only move a house up, down, left, or right. 
Of course, if it does not hit the wall! (The red lines are walls and it's impossible to cross them.) 


The heavy weight chained to the gang's foot not only made it possible for him to climb the wall, which even disturbed him through the houses of the map.


Steve first you have to cling to the keys. 


As you can see in the map of [prision](https://github.com/rabieifk/Prison-Break_Machine_Learning/blob/master/Prison.png), there are two keys in the prison that open both solo doors. 


Steve will need to take one of these keys and then go to the detained detainee in solitary confinement.


In some houses, the camera path is installed; if Steve stays in these homes, his image is recorded and tortured for being roamed in prison.


As you see in the [map](https://github.com/rabieifk/Prison-Break_Machine_Learning/blob/master/Prison.png) , the guardianship is being guarded. 
If Steve enters a house from the table where the guards are located, 
he will be arrested and exiled to the detained imprisoned until the end of his life and will no longer have the opportunity of this honorable profession!


The guards are at any moment with equal probability in one of the four houses marked with the * sign.

In order to avoid looping and confusing agent,reward of -1 should be considered for each transfer. The policy to find between 4 direction for Steve is [boltzmann](https://en.wikipedia.org/wiki/Boltzmann_distribution) policy at first with high temperature and then reduce the parameter.

When lambda is zero, [SARSA](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) and [Q-learning](https://en.wikipedia.org/wiki/Q-learning) are the same and just see the next state, if lamda is considered one, the result is the same as [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_algorithm) method which wait untill the end of episod.

There is a tradeoff between speed and performance in such problems. This tradeoff can be controled by lambda.

The best result is for monte Carlo but with most processing power and delay. 

With the policy itaration, agent can find the optimal policy.


