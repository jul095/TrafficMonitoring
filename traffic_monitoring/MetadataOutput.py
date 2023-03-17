#  ****************************************************************************
#  @ExtractTrajectories.py
#
#  Write Metadata about the video in a Json file
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import json
from json.decoder import JSONDecodeError
from os import error
from pathlib import Path

class MetadataOutput:

    def __init__(self, outputPath):
        self.outputPath = outputPath
        self.data = {}

    """Stores a Dictonary with all given outputParameter. outputParameter should be a Dictonary"""
    def extendOutput(self, outputParameter):
        self.data.update(outputParameter)
        print(self.data)

    """Writes outputParameter in a Json file. outputParameter should be a Dictonary"""
    def writeOutput(self):
        dataFromJson = {}
        outputFile = Path(self.outputPath)
        try: 
            if outputFile.is_file() == True:
                with open(self.outputPath, "rt") as file:
                    dataFromJson = json.load(file)

        except JSONDecodeError: #data is not serializable
            dataFromJson = {}

        self.data.update(dataFromJson)
        with open(self.outputPath, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)