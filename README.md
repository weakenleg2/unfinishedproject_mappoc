# Learning-Resource-Aware Communication and Control for Multiagent Systems
Repository for the master's thesis [Learning-Resource-Aware Communication and Control for Multiagent Systems](https://uu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=7&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&query=&language=sv&pid=diva2%3A1767669&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=2581)

## Repository Structure
```custom_envs/``` - Modified Multi-Particle Environment used in the thesis.

```maddpg/``` - Root folder for the MADDPG implementation.
```mappo/algorithms``` - Root folder for the MAPPO implementation.

## Training the agents
All hyperparameters and training curves can be found in the thesis. The default configurations of the algorithms should be similar to the ones presented in the thesis, but I would strongly suggest to double check the important parameters such as ```--n_rollout_threads``` and ```--n_trajectories``` before running the code.
### Steup
Install the components of this repository

```pip install -e```

### MADDPG
```python scripts/train_ra_maddpg.py```

Command line arguments for tweaking hyperparameters can be found in ```scripts/train_ra_maddpg.py```.

### MAPPO 
```python scripts/train_mappo.py```

Command line arguments for tweaking hyperparameters can be found in ```mappo/config.py```.



## Demo
### Centralized Critic
[https://www.youtube.com/watch?v=wPtRb_eRSyM](https://www.youtube.com/watch?v=wPtRb_eRSyM)

### Independent Agents
[https://www.youtube.com/watch?v=QUFTfX0G-n0](https://www.youtube.com/watch?v=QUFTfX0G-n0)

## Credits

The MADDPG and MAPPO implementations are based on the following repositories:

MADDPG: [https://github.com/shariqiqbal2810/maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch)

MAPPO: [https://github.com/marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)

