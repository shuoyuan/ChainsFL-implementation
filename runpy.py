import os
import sys
import json

taskQuery = os.popen(r"./taskQuery.sh")
taskInfoR = taskQuery.read()
taskQuery.close()
taskInfo = json.loads(taskInfoR)
users = taskInfo['usersList']

print (taskInfo['taskID'])
print (len(users))