### \#Exploration: A Study of Count-based Exploration for Deep Reinforcement Learning 
Haoran Tang*, Rein Houthooft*, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel<br/> 
(http://papers.nips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning)

The follwing command trains an agent on Montezuma's Revenge, as described in Section 5.3 of the Supplementary Material<br/>  (http://papers.nips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning-supplemental.zip)<br/>
`python sandbox/haoran/hashing/bonus_trpo/launchers/exp-021e2.py`

Please be reminded that training outcomes depend dramatically on the random seeds. The script above usually gives a final score among 400, 500, 2500, 6500, and other values, though the random seed `2000` is already chosen to achieve a score of 6500.

The following command simulates a well-trained agent that gets 6500 points.<br/>
`python scripts/sim_policy.py  tmp/success.pkl`

Please install Thenao and Lasagne via <br/>
`pip install theano` <br/>
`pip install https://github.com/Lasagne/Lasagne/archive/master.zip`
