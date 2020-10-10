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
  echo "      - 'query'"
  echo "    [Variables]"
  echo "    for 'release' taskID epoch status usersFrac"
  echo "    for 'local' modelFileTrainedLocally deviceID taskID currentEpoch"
  echo "    for 'global' globalModelFileAggregated taskID currentEpoch status"
  echo "    for 'query' queryInfo"
  echo " Examples:"
  echo "  interRun.sh release task1234 10 start 0.1"
  echo "  interRun.sh local device00010 task1234 2 fileHash"
  echo "  interRun.sh aggregated task1234 2 training fileHash"
  echo "  interRun.sh query task1234"
}

# prepending $PWD/../bin to PATH to ensure we are picking up the correct binaries
# this may be commented out to resolve installed version of tools if desired
export FabricL=/home/shawn/Documents/fabric-samples/test-network
export PATH=${FabricL}/../bin:$PATH
export FABRIC_CFG_PATH=$FabricL/../config/
# Turn on the tls
export CORE_PEER_TLS_ENABLED=true

# Set the environment variables
setEnvironments() {
  ORG=$1
  if [ $ORG -eq 1 ]; then
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    export CORE_PEER_ADDRESS=localhost:7051                                                                       
  elif [ $ORG -eq 2 ]; then                                                                                       
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt 
    export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
    export CORE_PEER_ADDRESS=localhost:9051
  else  
    echo "================== ERROR !!! ORG Unknown =================="                                            
  fi
}

# Release the FL task from fabric
function taskRelease() {
  ORG=$1
  setEnvironments $ORG
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${FabricL}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"taskRelease\",\"{\\\"taskID\\\":\\\"$2\\\",\\\"epoch\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"usersFrac\\\":$5}\"]}'"
  eval ${invoke}
}

# Publish the global model file aggregated in current epoch
function aggModelPub() {
  ORG=$1
  setEnvironments $ORG
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${FabricL}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"${2}\",\"{\\\"epoch\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"paras\\\":\\\"$5\\\"}\"]}'"
  eval ${invoke}
}

# Publish the local model file trained in current epoch
function localModelPub() {
  ORG=$1
  setEnvironments $ORG
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${FabricL}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${FabricL}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"${2}\",\"{\\\"taskID\\\":\\\"$3\\\",\\\"epoch\\\":$4,\\\"paras\\\":\\\"${5}\\\"}\"]}'"
  eval ${invoke}
}

# Query the info from the chaincode
function devQuery(){
  ORG=$1
  setEnvironments $ORG
  query="peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"${2}\"]}'"
  eval ${query}
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
  devQuery 1 'taskRelease'
elif [ "${MODE}" == "local" ]; then
  localModelPub 1 $1 $2 $3 $4
elif [ "${MODE}" == "aggregated" ]; then
  aggModelPub 1 $1 $2 $3 $4
  sleep 2
  devQuery 1 $1
elif [ "${MODE}" == "query" ]; then
  devQuery 1 $1
else
  printHelp
  exit 1
fi
