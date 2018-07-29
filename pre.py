from random import shuffle


def writeFile(filename, sentences):
    with open(filename, 'w') as fOut:
        for line in sentences:
            fOut.write(line)
            fOut.write("\n")
            
negSentences = []
posSentences = []

for line in open('/Users/pedroalonso/Documents/Python/negative.txt', 'r'):
    line = "0\t"+line.strip()
    negSentences.append(line)
    
for line in open('/Users/pedroalonso/Documents/Python/positive.txt'):
    line = "1\t"+line.strip()
    posSentences.append(line)
    

# print(negSentences, "negSentences")
# print(posSentences, "posSentences")

shuffle(negSentences)
shuffle(posSentences)

# print(negSentences, "negSentences shuffled")
# print(posSentences, "posSentences shuffled")
    
n = len(negSentences)    
trainSplit = (0, int(0.5*n))
devSplit = (trainSplit[1], trainSplit[1]+int(0.25*n))
testSplit = (devSplit[1], n)

print(trainSplit, "trainSplit")
print(testSplit, "testSplit")
print(devSplit, "devSplit")

train = negSentences[trainSplit[0]:trainSplit[1]] + posSentences[trainSplit[0]:trainSplit[1]]
dev = negSentences[devSplit[0]:devSplit[1]] + posSentences[devSplit[0]:devSplit[1]]
test = negSentences[testSplit[0]:testSplit[1]] + posSentences[testSplit[0]:testSplit[1]]

print(train, "train")
# print(dev, "dev")
# print(test, "test")

writeFile('train.txt', train)
writeFile('dev.txt', dev)
writeFile('test.txt', test)
