#!/bin/bash
# ******************************************************
# Filename     : allInOne.sh
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-09 16:11
# Description  : 
# ******************************************************

# Print the usage message
function printHelp() {
  echo "Usage: "
  echo "  allInOne.sh <Mode> [Variables]"
  echo "    <Mode>"
  echo "      - 'release'"
  echo "      - 'local'"
  echo "      - 'global'"
  echo "    [Variables]"
  echo "    for 'release' taskID epochs status usersFrac"
  echo "    for 'local' modelFileTrainedLocally deviceID taskID currentEpoch"
  echo "    for 'global' globalModelFileAggregated taskID currentEpoch status"
  echo " Examples:"
  echo "  allInOne.sh release task1234 10 start 0.1"
  echo "  allInOne.sh local device00010parameter.pkl device1234 task1234 2"
  echo "  allInOne.sh global device00010parameter.pkl task1234 2 training"
}

# prepending $PWD/../bin to PATH to ensure we are picking up the correct binaries
# this may be commented out to resolve installed version of tools if desired
export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/
# Turn on the tls
export CORE_PEER_TLS_ENABLED=true

# Set the environment variables
setEnvironments() {
  ORG=$1
  if [ $ORG -eq 1 ]; then
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=localhost:7051                                                                       
  elif [ $ORG -eq 2 ]; then                                                                                       
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
    export CORE_PEER_ADDRESS=localhost:9051
  else  
    echo "================== ERROR !!! ORG Unknown =================="                                            
  fi
}

# Release the FL task from fabric
function taskRelease() {
  set -x
  ORG=$1
  setEnvironments $ORG
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"taskRelease\",\"{\\\"taskID\\\":\\\"$2\\\",\\\"epochs\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"usersFrac\\\":$5}\"]}'"
  eval ${invoke}
  set +x
}

# Publish the global model file aggregated in current epoch
function globParInvoke() {
  set -x
  ORG=$1
  setEnvironments $ORG
  fileHash=$(ipfs add ${2} | tr ' ' '\n' | grep Qm)
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"${3}\",\"{\\\"epoch\\\":\\\"$4\\\",\\\"status\\\":$5,\\\"paras\\\":\\\"${fileHash}\\\"}\"]}'"
  eval ${invoke}
  set +x
  return 1
}

# Publish the local model file trained in current epoch
function parInvoke() {
  set -x
  ORG=$1
  setEnvironments $ORG
  fileHash=$(ipfs add ${2} | tr ' ' '\n' | grep Qm)
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"${3}\",\"{\\\"taskID\\\":\\\"$4\\\",\\\"epochs\\\":$5,\\\"paras\\\":\\\"${fileHash}\\\"}\"]}'"
  eval ${invoke}
  set +x
  return 1
}

# Query the info from the chaincode
function devQuery(){
  set -x
  query="peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"${1}\"]}'"
  eval ${query}
  set +x
  return 1
}

# Parse commandline args

## Parse mode

if [[ $# -lt 1 ]] ; then
  printHelp
  exit 0
else
  MODE=$1
  shift
fi

if [ "${MODE}" == "release" ]; then
  taskRelease 1 $1 $2 $3 $4
  sleep 2
  devQuery taskRelease
elif [ "${MODE}" == "local" ]; then
  parInvoke 1 $1 $2 $3 $4
  sleep 2
  devQuery $2
elif [ "${MODE}" == "global" ]; then
  globParInvoke 1 $1 $2 $3 $4
  sleep 2
  devQuery $2
else
  printHelp
  exit 1
fi
