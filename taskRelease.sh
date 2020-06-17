#!/bin/sh
# ******************************************************
# Filename     : invokeParameters.sh
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-04 17:42
# Description  : 
# ******************************************************

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

taskRelease() {
  ORG=$1
  setEnvironments $ORG
  set -x
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"taskRelease\",\"{\\\"taskID\\\":\\\"$2\\\",\\\"epochs\\\":$3,\\\"status\\\":\\\"$4\\\",\\\"usersFrac\\\":$5}\"]}'"
  eval ${invoke}
  set +x
}

taskRelease 1 $1 $2 $3 $4