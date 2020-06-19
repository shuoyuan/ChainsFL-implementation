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

parInvoke() {
  ORG=$1
  setEnvironments $ORG
  fileHash=$(ipfs add ${2} | tr ' ' '\n' | grep Qm)
  set -x
  invoke="peer chaincode invoke -o localhost:7050 --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc -c '{\"Args\":[\"set\",\"${3}\",\"{\\\"taskID\\\":\\\"$4\\\",\\\"epochs\\\":$5,\\\"paras\\\":\\\"${fileHash}\\\"}\"]}'"
  eval ${invoke}
  set +x
  return 1
}

devQuery(){
  query="peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"${1}\"]}'"
  eval ${query}
  return 1
  }

parInvoke 1 $1 $2 $3 $4
sleep 2
devQuery $2

