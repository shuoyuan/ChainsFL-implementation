#!/bin/bash
# ******************************************************
# Filename     : test.sh
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-08 19:07
# Description  : 
# ******************************************************

export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/

export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051

taskStatus=$(peer chaincode query -C mychannel -n sacc -c '{"Args":["get","taskRelease"]}' | jq ".status")
taskID=$(peer chaincode query -C mychannel -n sacc -c '{"Args":["get","taskRelease"]}' | jq ".taskID")
usersList=$(peer chaincode query -C mychannel -n sacc -c '{"Args":["get","taskRelease"]}' | jq ".usersList")

while :
do
    echo $taskID
    if [ ${taskStatus} == "done" ]
    then
        echo $taskStatus
        break
    else

    fi
done

