import numpy as np
import sys,os
import configparser

class InputFile():
    """Class for packaging all input/config file options together. 

    **Inputs**

    ----------
    inputfilename : string 
        String specifying the desired inputfile name.     

    **Options**

    ----------

    ==============================  ============  ===============================================
    option                          type          description
    ==============================  ============  ===============================================
    dimension_theta                 int           - Dimension of the model uncertainty space.
    distribution_theta              string        - Specifier for the distribution.
    data_file_type                  string        - Type of data specified in data_file.
                                                  - Supported options: csv, npy.
    structure_function              string        - Structure function specifier.
                                                  - Supported options: fourier
    dimensions                      int           - Number of physical dimensions of data.
                                                  - Supported options: 2, 3.
    output_file                     string        - Full path to output file name (without datatype extension)
    ==============================  ============  ===============================================
    """

    def __init__(self, inputfile):
        config    = configparser.ConfigParser()
        config.read(inputfile)
        keys,vals = self.assert_everything_included(config)
        self.set_options(keys,vals)
        self.convert_strings_to_appropriate_datatypes()

    def assert_everything_included(self,config):
        # Check that correct section header is used
        self.inputtype = config.sections()[0]
        # Get keys
        keys = [k for k in config[self.inputtype]]
        vals = [config[self.inputtype][v] for v in keys]
        self.set_necessary_keys()
        # Check that all necessary keys are included
        for k in self.necessary_keys:
            try:
                assert( any(k in s for s in keys) )
            except AssertionError as e:
                e.args += ('The following necessary argument was not specified in the input file: ' + k,)
                raise
        return keys,vals

    def set_options(self,keys,vals):
        # Set all options
        for i,k in enumerate(keys):
            setattr(self,k,vals[i])

    def convert_strings_to_appropriate_datatypes(self):
        self.dimensions = int(self.dimensions)
        
    def set_necessary_keys(self):
        # Set list defining necessary keys for training
        self.necessary_keys = ['data_file', \
                               'data_file_type', \
                               'structure_function', \
                               'dimensions',\
                               'output_file']

    def printInputs(self):
        """
        Method to print all config options.
        """
        attrs = vars(self);
        print('\n');
        print("********************* INPUTS *********************")
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        print("**************************************************")
        print('\n');
