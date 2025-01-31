#!/usr/bin/env python3
# Python program to set the version.
##############################################

import re
import sys
import optparse


def fileProcess(name, lineFunction):
    filestream = open(name, 'r')
    if filestream.closed:
        print("file " + name + " not open.")
        return

    output = ""
    print("--- Processing " + name + " ---------")
    while 1:
        line = filestream.readline()
        if not line:
            break
        output += lineFunction(line)
    filestream.close()

    if not output:
        return  # basic error checking

    print("Writing file " + name)
    filestream = open(name, "w")
    filestream.write(output)
    filestream.close()


def echoInput(line):
    return line


parser = optparse.OptionParser("usage: %prog major minor build")
(options, args) = parser.parse_args()
if len(args) != 3:
    parser.error("incorrect number of arguments")

major = args[0]
minor = args[1]
build = args[2]
versionStr = major + "." + minor + "." + build

print("Setting dox,tinyxml2.h")
print("Version: " + major + "." + minor + "." + build)

#### Write the tinyxml.h ####


def engineRule(line):

    matchMajor = "static const int TIXML2_MAJOR_VERSION"
    matchMinor = "static const int TIXML2_MINOR_VERSION"
    matchBuild = "static const int TIXML2_PATCH_VERSION"

    if line[0:len(matchMajor)] == matchMajor:
        print("1)tinyxml2.h Major found")
        return matchMajor + " = " + major + ";\n"

    elif line[0:len(matchMinor)] == matchMinor:
        print("2)tinyxml2.h Minor found")
        return matchMinor + " = " + minor + ";\n"

    elif line[0:len(matchBuild)] == matchBuild:
        print("3)tinyxml2.h Build found")
        return matchBuild + " = " + build + ";\n"

    else:
        return line


fileProcess("tinyxml2.h", engineRule)


def macroVersionRule(line):

    matchMajor = "#define TINYXML2_MAJOR_VERSION"
    matchMinor = "#define TINYXML2_MINOR_VERSION"
    matchBuild = "#define TINYXML2_PATCH_VERSION"

    if line[0:len(matchMajor)] == matchMajor:
        print("1)macro Major found")
        return matchMajor + " " + major + "\n"

    elif line[0:len(matchMinor)] == matchMinor:
        print("2)macro Minor found")
        return matchMinor + " " + minor + "\n"

    elif line[0:len(matchBuild)] == matchBuild:
        print("3)macro Build found")
        return matchBuild + " " + build + "\n"

    else:
        return line


fileProcess("tinyxml2.h", macroVersionRule)

#### Write the dox ####


def doxRule(line):

    match = "PROJECT_NUMBER"

    if line[0:len(match)] == match:
        print("dox project found")
        return "PROJECT_NUMBER = " + major + "." + minor + "." + build + "\n"

    else:
        return line


fileProcess("dox", doxRule)

#### Write the CMakeLists.txt ####


def cmakeRule(line):

    matchVersion = "project(tinyxml2 VERSION"

    if line[0:len(matchVersion)] == matchVersion:
        print("1)tinyxml2.h Major found")
        return matchVersion + " " + major + "." + minor + "." + build + ")\n"

    else:
        return line


fileProcess("CMakeLists.txt", cmakeRule)


def mesonRule(line):
    match = re.search(r"(\s*version) : '(\d+.\d+.\d+)',", line)
    if match:
        print("1)meson.build version found.")
        return "{} : '{}.{}.{}',\n".format(match.group(1), major, minor, build)
    return line


fileProcess("meson.build", mesonRule)

print("Release note:")
print('1. Build.   g++ -Wall -DTINYXML2_DEBUG tinyxml2.cpp xmltest.cpp -o gccxmltest.exe')
print('2. Commit.  git commit -am"setting the version to ' + versionStr + '"')
print('3. Tag.     git tag ' + versionStr)
print('   OR       git tag -a ' + versionStr + ' -m [tag message]')
print('Remember to "git push" both code and tag. For the tag:')
print('git push origin [tagname]')
