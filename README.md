# ChainsFL
Implementation of ChainsFL.

## Requirments

- [Hyperledge Fabric 2.1](https://hyperledger-fabric.readthedocs.io/en/release-2.1/test_network.html#before-you-begin)
- Python3
- Pytorch
- Torchvision
- [IPFS](https://docs.ipfs.io/install/command-line/#official-distributions)

**Attention**: The configs of Fabric in `./commonComponent/interRun.sh` should be modified to access the Fabric deployed above. Besides, this file should be authorized with the right of *writer/read/run*.

## Deployment of DAG

The DAG could be deployed on a personal computer or the cloud server.

Copy all the files of this repository to the PC or cloud server, and then run following commands in the root path of this repository.

```
cd dagMainChain
python serverRun.py
``` 

## Run one shard

The shard also could be deployed on a personal computer or the cloud server.

Copy all the files of this repository to the deployment location, and modify `line 466` of `dagMainChain/clientRun.py` for the real address of the DAG server deployed above. Then run following commands in the root path of this repository.

```
# run the DAG client
cd dagMainChain
python clientRun.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --num_channels 1

# run the FL task
cd federatedLearning
python main_fed_local.py --epochs 1 --frac 0.1 --gpu -1 --model cnn --num_channels 1
```

The details of these parameters could be found in file `federatedLearning/utils/options.py`. 
It should be noted that the `--epochs` configured in command with `clientRun.py` represents the number of rounds run in each shard.
And the `--epochs` configured in command with `main_fed_local.py` represents the number of epochs run on each local device.

## Run multiple shards

Similar to the above, copy all the files of this repository and then modify the files and execute the commands presented above.

Besides, the para of `nodeNum` in `line 58` of `dagMainChain/clientRun.py` indicates the shard index which should be modified.

## Acknowledgments

Acknowledgments give to [shaoxiongji](https://github.com/shaoxiongji/federated-learning) and [AshwinRJ](https://github.com/AshwinRJ/Federated-Learning-PyTorch) for the basic codes of the FL module.

