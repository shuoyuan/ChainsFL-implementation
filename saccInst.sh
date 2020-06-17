#!/bin/bash
# ******************************************************
# Filename     : saccInst.sh
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-07 20:53
# Description  : 
# ******************************************************


export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/
# Turn on the tls
export CORE_PEER_TLS_ENABLED=true

peer version

pushd ../chaincode/sacc
GO111MODULE=on go mod vendor
popd

# Verify the result of previous command
verifyResult() {
  if [ $1 -ne 0 ]; then
    echo "!!!!!!!!!!!!!!! "$2" !!!!!!!!!!!!!!!!"
    echo
    exit 1
  fi
}

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

packageChaincode() {
  ORG=$1
  setEnvironments $ORG
  set -x
  peer lifecycle chaincode package sacc.tar.gz --path ../chaincode/sacc --lang golang --label sacc_1 >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Chaincode packaging on peer0.org${ORG} has failed" 
  echo "===================== Chaincode is packaged on peer0.org${ORG} ===================== "
  echo
}

installChaincode() {
  ORG=$1
  echo "================== Begin the installation on ORG${ORG} =================="
  set -x
  setEnvironments $ORG
  peer lifecycle chaincode install sacc.tar.gz >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Chaincode installation on peer0.org${ORG} has failed"
  echo "===================== Chaincode is installed on peer0.org${ORG} ===================== "
  echo
}

queryInstalled() {
  ORG=$1
  set -x
  setEnvironments $ORG 
  peer lifecycle chaincode queryinstalled >& log.txt
  res=$?
  set +x
  cat log.txt
    CC_PACKAGE_ID=$(sed -n "/sacc_1/{s/^Package ID: //; s/, Label:.*$//; p;}" log.txt)
  verifyResult $res "Query installed on peer0.org${ORG} has failed" 
  echo PackageID is ${CC_PACKAGE_ID}
  echo "===================== Query installed successful on peer0.org${ORG} on channel ===================== "
  echo
}

approveForMyOrg() {
  ORG=$1
  set -x
  setEnvironments $ORG
  echo $CC_PACKAGE_ID
  peer lifecycle chaincode approveformyorg -o localhost:7050 \
	--ordererTLSHostnameOverride orderer.example.com \
	--channelID mychannel --name sacc --version 1.0 --init-required \
	--package-id $CC_PACKAGE_ID --sequence 1 --tls true \
	--cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Chaincode definition approved on peer0.org${ORG} on channel '$CHANNEL_NAME' failed"
  echo "===================== Chaincode definition approved on peer0.org${ORG} ===================== "
  echo
}

## at first we package the chaincode
packageChaincode 1

## Install chaincode on peer0.org1 and peer0.org2
echo "Installing chaincode on peer0.org1..."
installChaincode 1
echo "Install chaincode on peer0.org2..."
installChaincode 2

## query whether the chaincode is installed
queryInstalled 1

## approve the definition for org1
approveForMyOrg 1

## now approve also for org2
approveForMyOrg 2

echo "===================== Check the status of approve ===================== "
set -x
peer lifecycle chaincode checkcommitreadiness --channelID mychannel --name sacc --version 1.0 --init-required --sequence 1 --tls true \
	--cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --output json
set +x
echo

echo "===================== Commit the approved chaincode ===================== "
set -x
peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID mychannel --name sacc --version 1.0 --sequence 1 --init-required --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
set +x
echo

echo "===================== Check the status of the chaincode commit ===================== "
set -x
peer lifecycle chaincode querycommitted --channelID mychannel --name sacc --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem
set +x
echo

echo "===================== Init the chaincode ===================== "
setEnvironments 1
set -x
peer chaincode invoke -o localhost:7050 --tls true --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt -C mychannel -n sacc --isInit -c '{"Args":["deviceID0000","hashValueTest"]}'
set +x
echo

sleep 3 
echo "===================== Query test ===================== "
set -x
peer chaincode query -C mychannel -n sacc -c '{"Args":["get","deviceID0000"]}'
set +x
