import os
import argparse

def adaptScp(scpFile, output):
    result = {}
    counter = 0
    with open(output, "w") as outputFile:
        with open(scpFile) as f:
            for line in f:
                splitted = line.split("=")
                if len(splitted) != 2:
                    raise Exception("Line with invalid format '%s'" % line)
                key = splitted[0].strip()
                if key.endswith(".mfc"):
                    key = key[:-4]
                newKey = '"' + str(counter) + '"'
                result[key] = newKey
                outputLine = newKey + "=" + splitted[1]
                outputFile.write(outputLine)
                counter += 1
    return result

def adaptMlf(mlfFile, output, keys):
    previous = ""
    with open(output, "w") as outputFile:
        with open(mlfFile) as inputFile:
            for originalLine in inputFile:
                line = originalLine.strip()
                if previous == "." or previous == "#!MLF!#":
                    line = line.strip('"')
                    if line.endswith(".lab"):
                        line = line[:-4]

                    if line in keys:
                        line = keys[line] + "\n"
                        outputFile.write(line)
                    else:
                        outputFile.write(originalLine)
                else:
                    outputFile.write(originalLine)
                previous = line.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapts mlf and scp files to use logical keys of smaller length")
    parser.add_argument('--inputScpFile', help='Input scp file.', required=True)
    parser.add_argument('--inputMlfFile', help='Input mlf file that corresponds to the scp file.', required=True)
    parser.add_argument('--outputScpFile', help='Output scp file.', required=True)
    parser.add_argument('--outputMlfFile', help='Output mlf file.', required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.inputScpFile):
        print('Input scp file is invalid.')

    if not os.path.exists(args.inputMlfFile):
        print('Input mlf file is invalid.')

    keys = adaptScp(args.inputScpFile, args.outputScpFile)
    adaptMlf(args.inputMlfFile, args.outputMlfFile, keys)