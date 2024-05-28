"""
Contains all variables necessary to stream data
"""


# list of rigid body parts 
rigidBodyParts = ['Pelvis', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 
                      'LFArm', 'LHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 
                      'LIndex3', 'LMiddle1', 'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', 
                      'LPinky1', 'LPinky2', 'LPinky3', 'RShoulder', 'RUArm', 'RFArm', 'RHand', 'RThumb1', 
                      'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 'RMiddle2', 'RMiddle3', 
                      'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3', 'LThigh', 'LShin', 
                      'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']

bodyParts = ['Pelvis', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 
                      'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand',  'LThigh', 'LShin', 
                      'LFoot', 'RThigh', 'RShin', 'RFoot']


# dict of body parts, where each vector represents [importance = 1 (if important), (unit quaternion), color to plot]

rigidBodyPartsDict = {'Pelvis': [1,(0,0,1),'b'], 'Ab': [1,(0,0,1),'b'], 'Chest': [1,(0,-1,0),'b'], 'Neck': [1,(0,0,1),'g'], 'Head': [1,(0,-1,0),'g'], 'LShoulder': [1,(1,0,0),'r'], 'LUArm': [1,(1,0,0),'r'], 
                    'LFArm': [1,(1,0,0),'r'], 'LHand': [1,(1,0,0),'c'], 'LThumb1': [0], 'LThumb2': [0], 'LThumb3': [0], 'LIndex1': [0], 'LIndex2': [0], 
                    'LIndex3': [0], 'LMiddle1': [0], 'LMiddle2': [0], 'LMiddle3': [0], 'LRing1': [0], 'LRing2': [0], 'LRing3': [0], 
                    'LPinky1': [0], 'LPinky2': [0], 'LPinky3': [0], 'RShoulder': [1,(-1,0,0),'m'], 'RUArm': [1,(-1,0,0),'m'], 'RFArm': [1,(-1,0,0),'m'], 'RHand': [1,(-1,0,0),'y'], 'RThumb1': [0], 
                    'RThumb2': [0], 'RThumb3': [0], 'RIndex1': [0], 'RIndex2': [0], 'RIndex3': [0], 'RMiddle1': [0], 'RMiddle2': [0], 'RMiddle3': [0], 
                    'RRing1': [0], 'RRing2': [0], 'RRing3': [0], 'RPinky1': [0], 'RPinky2': [0], 'RPinky3': [0], 'LThigh': [1,(0,-1,0),'c'], 'LShin':  [1,(0,-1,0),'c'], 
                    'LFoot':  [1,(0,-1,0),'c'], 'LToe': [0], 'RThigh':  [1,(0,-1,0),'c'], 'RShin':  [1,(0,-1,0),'c'], 'RFoot':  [1,(0,-1,0),'c'], 'RToe': [0]}
  
simpleBodyParts = [0,1,2,3,4,5,6,7,8,24,25,26,27,43,44,45,47,48,49] # list of body parts to extract

#simpleBodyPartLabels = rigidBodyParts[simpleBodyParts]


renderingBodyParts = []
renderingBodyPartsIdxes = []
counter = 0
for bodyPart in rigidBodyPartsDict:
    if rigidBodyPartsDict[bodyPart][0] == 1 and counter in simpleBodyParts:
        renderingBodyParts.append(bodyPart)
        renderingBodyPartsIdxes.append(counter)
    counter += 1
quaternionsUnit = [rigidBodyPartsDict[bodyPart][1] for bodyPart in renderingBodyParts]
colourCode = [rigidBodyPartsDict[bodyPart][2] for bodyPart in renderingBodyParts]

rightHandIndex = 27