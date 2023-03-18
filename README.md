### \#Exploration: A Study of Count-based Exploration for Deep Reinforcement Learning 
Haoran Tang*, Rein Houthooft*, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel<br/> 
(http://papers.nips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning)

#### Prerequisites
Please install Thenao and Lasagne via <br/>
`pip install theano` (version 1.0.1) <br/>
`pip install https://github.com/Lasagne/Lasagne/archive/master.zip` (version 0.2.dev1) <br/>
If you see the message `WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS ` during experiments and observe in `htop` that all CPU utilization bars are in red, then BLAS is not properly installed and the experiment will run incredibly slow. [This link](https://github.com/Theano/Theano/issues/6532) suggests a solution: `sudo apt install libblas-dev
`.


#### Descriptions
* This code base is based on Rocky Duan's [RLLab framework](https://github.com/rll/rllab)
* The training algorithm is Parallel TRPO written in Theano. It has a compilation stage before training. During training, it iterates between collecting large batches of data and calling TRPO to update the policy parameters.
* Visualization: call `python rllab/viskit/frontend {log_dir}`. An example `log_dir` is `data/local/bonus-trpo-atari/exp-027/exp-027_20180420_170633_281191_freeway`.
* Simulation: `python scripts/sim_policy.py {log_dir}/itr_{n}.pkl`, where `n` is the training iteration.
* It is suggested to run these experiments on AWS. You can use the pre-built public AMI `ami-29a6954c` (20180420_static_hashing) and find all code in `~/hashing/rllab-private`. c4.8xlarge (18 cores) is the preferred instance type.

#### SimHash on Atari games
The following script reproduces the SimHash results in Table 1. <br/>
`python sandbox/haoran/hashing/bonus_trpo/launchers/exp-027.py` <br/>
Please be reminded that training outcomes depend dramatically on the random seeds. But statistically, SimHash performs better than without the hash bonuses, as illustrated in Table 7 of the [Supplementary Material](http://papers.nips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning-supplemental.zip). 

#### SmartHash on Montezuma's Revenge
The follwing script trains an agent on Montezuma's Revenge, as described in Section 5.3 of the [Supplementary Material](http://papers.nips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning-supplemental.zip)<br/>
`python sandbox/haoran/hashing/bonus_trpo/launchers/exp-021e2.py`

Please be reminded that training outcomes depend dramatically on the random seeds. The script above usually gives a final score among 400, 500, 2500, 6500, and other values, though the random seed `2000` is already chosen to achieve a score of 6500.

The following command simulates a well-trained agent that gets 6500 points.<br/>
`python scripts/sim_policy.py  example_data/montezuma_revenge_success.pkl`

