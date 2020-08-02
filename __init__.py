'''
 MAYO pipeline, from Pairet et al. 2020
    Copyright (C) 2020, Benoit Pairet

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
'''


import numpy as np
import vip_hci as vip


from mayo_hci.operators import *
from mayo_hci.algo_utils import *

from mayo_hci.mayonnaise import *

from mayo_hci.automatic_load_data import *
from mayo_hci.create_synthetic_data_with_disk_planet import *
from mayo_hci.GreeDS import *

