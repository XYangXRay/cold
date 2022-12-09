#!/usr/bin/env python3

import cold
import fire
import dxchange

def main(path, debug=False):
    file, comp, geo, algo = cold.config(path)
    # data = cold.loadsingle(file, id=1)
    # dxchange.write_tiff(data)
    data, ind = cold.load(file)
    pos, sig, scl = cold.decode(data, ind, comp, geo, algo, debug=True)
    dep, lau = cold.resolve(data, ind, pos, sig, geo, comp)
    shape = geo['detector']['shape']
    cold.saveimg('tmp/pos/pos', pos, ind, shape)
    cold.plotarr('tmp/sig/sig', sig, plots=False)
    cold.saveplt('tmp/dep/dep', dep, geo['source']['grid'])
    cold.saveimg('tmp/lau/lau', lau, ind, shape, swap=True)

# """Runs the reconstruction workflow given parameters 
# in a configuration file.

# Parameters
# ----------
# path: string
# Path of the YAML file with configuration parameters.

# scanpoint: int
# ID for the illumination scan point.

# scanframe: [int, int, int, int]
# Rectangular detector frame for the analysis.

# debug: bool
# If True, plots the fitted signals. 

# Returns
# -------
# None
# """

if __name__ == '__main__':
    fire.Fire(main)