#!/usr/bin/env python3
# File       : check_coverage.py
# Description: Python script that extracts the test coverage from XML file and exits successfully if benchmark is achieved

BENCHMARK = 0.9

import xml.etree.ElementTree as ET
import sys

def parseXML(xmlfile):
  
    # create element tree object
    tree = ET.parse(xmlfile)
  
    # get root element
    dict = tree.getroot()
    coverage = dict.attrib['line-rate']
    print(coverage)
    if float(coverage) > BENCHMARK:
        sys.exit(0)
    else:
        sys.exit(1)
      

if __name__ == "__main__":
    parseXML('coverage.xml')
    
