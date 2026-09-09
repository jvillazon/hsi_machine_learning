from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2

import tqdm

import tifffile
import zarr




CONFIG = {
    'base_dir': r'/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/Dataset_Root',
    
    'values_dir': 'values',
    'img_extensions' : ('.tif', '.ome.tiff', '.tiff'),

    'mask_dir': 'fib-masks',
    'subgroups' : {
      "C1": "Outer Cortex",
      "C2": "Middle Cortex",
      "C3": "Juxtamedullary Cortex",
      "CM": "Outer Stripe",
      "M": "Inner Stripe",
      "P": "Inner Medulla"
    },

    'int_dilations' : 2,
    'col_dilations' : 10,
    'min_instance_size' : 100,
}

MASK_DICTS = {
    "CFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "CFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "CFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "CFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "CFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "CFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "CFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "CFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "CFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "CFIB"
        }   
    },
    "CMFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "CMFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "CMFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "CMFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "CMFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "CMFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "CMFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "CMFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "CMFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "CMFIB"
        }
    },
    "CMYOF_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "CMYOF"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "CMYOF"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "CMYOF"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "CMYOF"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "CMYOF"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "CMYOF"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "CMYOF"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "CMYOF"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "CMYOF"
        }
    },
    "dFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "dFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "dFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "dFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "dFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "dFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "dFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "dFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "dFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "dFIB"
        }
    },
    "dIMFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "dIMFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "dIMFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "dIMFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "dIMFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "dIMFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "dIMFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "dIMFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "dIMFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "dIMFIB"
        }
    },
    "dOMFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "dOMFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "dOMFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "dOMFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "dOMFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "dOMFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "dOMFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "dOMFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "dOMFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "dOMFIB"
        }
    },
    "IMFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "IMFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "IMFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "IMFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "IMFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "IMFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "IMFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "IMFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "IMFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "IMFIB"
        }
    },
    "IMpvFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "IMpvFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "IMpvFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "IMpvFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "IMpvFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "IMpvFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "IMpvFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "IMpvFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "IMpvFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "IMpvFIB"
        }
    },
    "OMFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "OMFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "OMFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "OMFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "OMFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "OMFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "OMFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "OMFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "OMFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "OMFIB"
        }
    },
    "pvFIB_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "pvFIB"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "pvFIB"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "pvFIB"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "pvFIB"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "pvFIB"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "pvFIB"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "pvFIB"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "pvFIB"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "pvFIB"
        }
    },
    "pvMYOF_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "pvMYOF"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "pvMYOF"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "pvMYOF"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "pvMYOF"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "pvMYOF"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "pvMYOF"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "pvMYOF"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "pvMYOF"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "pvMYOF"
        }
    },
    "CMMYOF_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "CMMYOF"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "CMMYOF"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "CMMYOF"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "CMMYOF"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "CMMYOF"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "CMMYOF"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "CMMYOF"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "CMMYOF"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "CMMYOF"
        }
    },
    "IMpvMYOF_cluster":{
        0: {
            'slice_name': "Cluster26",
            'unit': "Interstitial cells | IGFBP7",
            'group': "IMpvMYOF"
            },
        1: {
            'slice_name': "Cluster18",
            'unit': "Endothelium | CD31",
            'group': "IMpvMYOF"
            },
        2: {
            'slice_name': "Cluster19",
            'unit': "Endothelium | CD31, Podo",
            'group': "IMpvMYOF"
            },
        3: {
            'slice_name': "Cluster20",
            'unit': "Pericytes | CD90, CD31(-)",
            'group': "IMpvMYOF"
            },
        4: {
            'slice_name': "Cluster21",
            'unit': "VSMCs | aSMA",
            'group': "IMpvMYOF"
            },
        5: {
            'slice_name': "Cluster22",
            'unit': "Immune | CD20",
            'group': "IMpvMYOF"
            },
        6: {
            'slice_name': "Cluster23",
            'unit': "Immune - MP | CD45, CD206",
            'group': "IMpvMYOF"
        },
        7: {
            'slice_name': "Cluster24",
            'unit': "Immune - T | CD3, CD4",
            'group': "IMpvMYOF"
        },
        8: {
            'slice_name': "Cluster25",
            'unit': "Immune - T | CD3, CD8",
            'group': "IMpvMYOF"
        }
    }
}

# UNIT_DICT = {
#     "Cluster1": "Glomeruli | Podo, CD31",
#     "Cluster2": "PT | LRP2",
#     "Cluster3": "PT | LRP2, AQP1, CD90",
#     "Cluster4": "PT | THY1",
#     "Cluster5": "PT | PROM1/VCAM1",
#     "Cluster6": "TDL | AQP1, Cyto8 (N1)",
#     "Cluster7": "TDL | AQP1, Cyto8 (N2)",
#     "Cluster8": "TAL | NaK/UMOD",
#     "Cluster9": "TAL | UMOD",
#     "Cluster10": "TAL | UMOD",
#     "Cluster11": "DCT | SLC12a3",
#     "Cluster12": "DN | Cyto8",
#     "Cluster13": "DN | Cyto8",
#     "Cluster14": "DN | Cyto8",
#     "Cluster15": "DN | Cyto8, Bcat",
#     "Cluster16": "DN | Cyto8, Bcat, Ecad",
#     "Cluster17": "DN | Cyto8, Ecad",
#     "Cluster18": "Endothelium | CD31",
#     "Cluster19": "Endothelium | CD31, Podo",
#     "Cluster20": "Pericytes | CD90, CD31(-)",
#     "Cluster21": "VSMCs | aSMA",
#     "Cluster22": "Immune | CD20",
#     "Cluster23": "Immune - MP | CD45, CD206",
#     "Cluster24": "Immune - T | CD3, CD4",
#     "Cluster25": "Immune - T | CD3, CD8",
#     "Cluster26": "Interstitial cells | IGFBP7"
# }

VAL_DICTS = {
  'Negative_Ion': {
    0: {'slice_name': "LPE 18:0 (-H)"},
    1: {'slice_name': "Unknown (N1)"},
    2: {'slice_name': "LPI 18:0 (-H)"},
    3: {'slice_name': "SM 18:1;2O/16:0 (-CH3)"},
    4: {'slice_name':"PA 36:1 (-H)"},
    5: {'slice_name':"PE 16:0_18:2 (-H)"},
    6: {'slice_name':"PE 16:0_18:0 (-H)"},
    7: {'slice_name':"PE 16:0_20:4 (-H)"},
    8: {'slice_name':"PE 16:0_22:6 (-H)"},
    9: {'slice_name':"PE 18:1_20:4 (-H)"},
    10: {'slice_name':"PE 18:1_20:3 (-H)"},
    11: {'slice_name':"PS 16:0_20:4 (-H)"},
    12: {'slice_name':"PS 18:0_18:1 (-H)"},
    13: {'slice_name':"PS 18:0_20:4 (-H)"},
    14: {'slice_name':"PG 18:1_22:6 (-H)"},
    15: {'slice_name':"HexCer 18:0;3O/23:0;(2OH) (-H)"},
    16: {'slice_name':"PI 34:1 (-H)"},
    17: {'slice_name':"PI 16:0_18:0 (-H)"},
    18: {'slice_name':"PI 16:0_20:4 (-H)"},
    19: {'slice_name':"PI 18:0_18:2 (-H)"},
    20: {'slice_name':"PI 18:0_18:1 (-H)"},
    21: {'slice_name':"SHexCer 40:1;3O (-H)"},
    22: {'slice_name':"PI 18:1_20:3 (-H)"},
    23: {'slice_name':"SHexCer 41:1;3O (-H)"},
    24: {'slice_name':"SM(d34:1) (+DMACA)"},
    25: {'slice_name':"PC 46:9 (-CH3)"},
    26: {'slice_name':"SHexCer 42:2;3O (-H)"},
    27: {'slice_name':"SHexCer 42:1;3O (-H)"},
    28: {'slice_name':"SHexCer 42:0;3O (-H)"},
    29: {'slice_name':"PE 49:9 (-H)"},
    30: {'slice_name':"PC 48:9 (-CH3)"},
    31: {'slice_name':"SHexCer 50:0;2O (-H)"},
    32: {'slice_name':"CL 72:8 (-H)"},
    33: {'slice_name':"CL 74:11 (-H)"},

  },
  'Positive_Ion': {
    0: {'slice_name': "Unknown (P1)"},
    1: {'slice_name': "Unknown (P2)"},
    2: {'slice_name':"CAR 18:2 (+H)"},
    3: {'slice_name':"CAR 18:1 (+H)"},
    4: {'slice_name':"CAR 18:0 (+H)"},
    5: {'slice_name':"Cer 17:2;2O/8:0;O (+Na)"},
    6: {'slice_name':"LPC 16:0 (+H)"},
    7: {'slice_name':"Unknown (P3)"},
    8: {'slice_name':"SM 34:1;2O (+H)"},
    9: {'slice_name':"Unknown (P4)"},
    10: {'slice_name':"PC 32:0 (+H)"},
    11: {'slice_name':"PC 34:2 (+H)"},
    12: {'slice_name':"SM 38:1;20 (+H)"},
    13: {'slice_name':"PC 16:0_18:1 (+H)"},
    14: {'slice_name':"PC 35:4 (+H)"},
    15: {'slice_name':"PC 36:5 (+H)"},
    16: {'slice_name':"PC 36:3 (+H)"},
    17: {'slice_name':"PC 18:0_18:1 (+H)"},
    18: {'slice_name':"PC O-38:5 (+H)"},
    19: {'slice_name':"PC 38:5 (+H)"},
    20: {'slice_name':"PC 18:0_20:4 (+H)"},
    21: {'slice_name':"SM 42:2;2O (+H)"},
    22: {'slice_name':"SM 42:1;2O (+H)"},
    23: {'slice_name':"SM 41:0;3O (+H)"},
    24: {'slice_name':"SM 42:0;3O (+H)"}
  },
}



METADATA = {
    'S24-K21_273_3' : {'condition': 'Reference'},
    'S21-K21_198_3' : {'condition': 'Reference'}
}

class MaskExtractor:
    def __init__(self, function=np.nanmean):
        # Contain directory of cell/unit id's based on centroid location
        # self.centroid_to_id = {}
        # self.next_id = 1
        self.results = []
        self._metadata_cache = {}
        self._current_mask_ref = None

        # Assign masking function (mean for ratio / intensity images and percentage for prediction images)
        self.func = function
    
    # def _get_instance_id(self, centroid):
    #     """Function for identifying cell/unit id based on centroids"""
    #     if centroid not in self.centroid_to_id:
    #         self.centroid_to_id[centroid] = self.next_id
    #         self.next_id += 1
    #     return self.centroid_to_id[centroid]

    def _pad_obj_slice(self, obj_slice, dilations, image_shape):
        """Function for padding object slice""" 
        padded_slices = []
        insert_slices = []
        for coord_slice, max_len in zip(obj_slice, image_shape):
            pad_start = max(0, coord_slice.start - dilations)
            pad_end = min(max_len, coord_slice.stop + dilations)
            padded_slices.append(slice(pad_start, pad_end))
            
            insert_start = coord_slice.start - pad_start
            insert_stop = coord_slice.stop - pad_start
            insert_slices.append(slice(insert_start, insert_stop))
            
        return tuple(padded_slices), tuple(insert_slices)

    def _build_instance_metadata(self, mask_2d, dilations, min_instance_size=0):
        """Precompute labeled instances, padded slices, and centroids for a single 2D mask."""
        if mask_2d.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask_2d.shape}")

        labels, num_features = ndi.label(mask_2d)
        if num_features == 0:
            return None

        slices = ndi.find_objects(labels)
        instance_metadata = []

        for label_idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            local_mask = labels[slc] == label_idx
            if min_instance_size and int(np.sum(local_mask)) < min_instance_size:
                continue
            
            local_rows, local_cols = np.nonzero(local_mask)
            global_rows = local_rows + slc[0].start
            global_cols = local_cols + slc[1].start
            global_centroid = (int(np.mean(global_rows)), int(np.mean(global_cols)))

            padded_slc, insert_slc = self._pad_obj_slice(
                obj_slice=slc,
                dilations=dilations,
                image_shape=mask_2d.shape
            )

            padded_mask_shape = tuple(s.stop - s.start for s in padded_slc)
            padded_mask = np.zeros(padded_mask_shape, dtype=bool)
            padded_mask[insert_slc] = local_mask

            if dilations > 0:
                dilated_mask = ndi.binary_dilation(padded_mask, iterations=dilations)
            else:
                dilated_mask = padded_mask

            instance_metadata.append({
                # 'instance_id': label_idx,
                # 'instance_label': f"{label_idx}",
                'label_idx': label_idx,
                'padded_slice': padded_slc,
                'instance_mask': dilated_mask,
                'centroid_x': global_centroid[1],
                'centroid_y': global_centroid[0],
            })

        return instance_metadata

    def _process_instance_values(self, instance_metadata, val_2d, mask_name, val_name, metadata):
        """Quantify a single value image using precomputed instance metadata."""
        val_2d_array = np.array(val_2d, dtype=float)
        for inst in instance_metadata:
            val_crop = val_2d_array[inst['padded_slice']]
            val_mask = val_crop[inst['instance_mask']]
            # total_pixel_count = int(val_mask.size)
            positive_mask = val_mask[(val_mask > 0) & np.isfinite(val_mask)]
            # positive_pixel_count = int(positive_mask.size)

            val_value = self.func(positive_mask) if int(positive_mask.size) > 0 else np.nan

            row_data = {
                # 'instance_id': inst['instance_id'],
                # 'instance_label': inst['instance_label'],
                'mask_name': mask_name,
                'val_name': val_name,
                'centroid_y': inst['centroid_y'],
                'centroid_x': inst['centroid_x'],
                # 'total_pixel_count': total_pixel_count,
                # 'positive_pixel_count': positive_pixel_count,
                # 'positive_pixel_fraction': (positive_pixel_count / total_pixel_count) if total_pixel_count else np.nan,
                'value': val_value
            }
            row_data.update(metadata)
            self.results.append(row_data)

    def process_image(self, mask_image, value_image, mask_dictionary, value_dictionary, dilations=2, min_instance_size=50, **metadata):
        """Global function for processing possible 3D masks and value images"""

        mask_metadata = mask_dictionary or {}
        value_metadata = value_dictionary or {}

        if self._current_mask_ref is not mask_image:
            self._metadata_cache = {}
            self._current_mask_ref = mask_image

        if value_image.ndim == 2:
            num_value_slices = 1
            value_shape = value_image.shape
        else:
            num_value_slices = value_image.shape[0]
            value_shape = value_image.shape[1:]

        num_mask_slices = 1 if mask_image.ndim == 2 else mask_image.shape[0]

        for mask_idx in range(num_mask_slices):
            mask_2d = mask_image if mask_image.ndim == 2 else mask_image[mask_idx]
            slice_mask_meta = mask_metadata.get(mask_idx, {})

            # Lookup name and fall back to index if nonexistent in dictionary
            m_name = slice_mask_meta.get('slice_name', f"Mask_{mask_idx}")

            cache_key = (metadata.get("tissue_id", "Tissue ID"), mask_dictionary[mask_idx].get('slice_name', f"Mask_{mask_idx}"), mask_dictionary[mask_idx].get('group',f'group_{mask_idx}'), value_shape, dilations, min_instance_size)
            if cache_key in self._metadata_cache:
                instance_metadata = self._metadata_cache[cache_key]
            else:
                if mask_2d.shape != value_shape:
                    mask_2d = cv2.resize(mask_2d.astype(np.uint8), (value_shape[1], value_shape[0]), interpolation=cv2.INTER_NEAREST)

                instance_metadata = self._build_instance_metadata(
                    mask_2d=mask_2d,
                    dilations=dilations,
                    min_instance_size=min_instance_size
                )
                self._metadata_cache[cache_key] = instance_metadata

            if not instance_metadata:
                continue
            
            for val_idx in range(num_value_slices):
                val_2d = value_image if value_image.ndim == 2 else value_image[val_idx]
                slice_value_meta = value_metadata.get(val_idx, {})

                # Lookup name again for value
                v_name = slice_value_meta.get('slice_name', f"Value_{val_idx}")

                # Combine all metadata into a single dictionary
                combined_metadata = {
                    **metadata,
                    **slice_mask_meta,
                    **slice_value_meta
                }

                # Process single 2D pair
                self._process_instance_values(
                    instance_metadata=instance_metadata,
                    val_2d=val_2d,
                    mask_name=m_name,
                    val_name=v_name,
                    metadata=combined_metadata
                )
    
    def return_dataframe(self):
        """Function for converting the aggregated results into a Pandas DataFrame"""
        return pd.DataFrame(self.results)
    
def _lazy_load_image(file_path, res_level=0):
    """Function for loading standard .tif images and pyramidal ome-tiffs memory efficiently"""
    store = tifffile.imread(file_path, aszarr=True)
    z = zarr.open(store, mode='r')

    if isinstance(z, zarr.hierarchy.Group):
        # if pyramidal ome-tiff
        return z[str(res_level)]
    else:
        # if standard tif
        return z 
    
def _parse_mask_basename(basename, subgroups):
    """Return cleaned unit name and optional subgroup from a mask basename."""
    mask_name = basename

    # Split mask name into constitutive parts given pattern "<prefix>-<unit_name>-<mask.ext>"
    mask_parts = mask_name.split('-')
    unit_name = mask_parts[-2]

    # Convert to lower case to be case insensitive
    subgroups_lower = {k.lower(): v for k,v in subgroups.items()}

    subgroup = None
    if "_" in unit_name and subgroups_lower:
        unit_parts = unit_name.split("_")

        if len(unit_parts) == 2:
            # Handle the specific 2-part logic
            match0 = subgroups_lower.get(unit_parts[0].lower())
            match1 = subgroups_lower.get(unit_parts[1].lower())
            
            if match0:
                subgroup = match0  # Maps to the dictionary value
                unit_name = unit_parts[1]
            elif match1:
                subgroup = match1  # Maps to the dictionary value
                unit_name = unit_parts[0]
        else:
            for i, part in enumerate(unit_parts):
                match = subgroups_lower.get(part.lower())
                if match:
                    subgroup = match  # Maps to the dictionary value
                    remaining = [p for j, p in enumerate(unit_parts) if i != j]
                    unit_name = "_".join(remaining)
                    break

    return unit_name, subgroup


if __name__ == "__main__":

    extractor = MaskExtractor()
    data_root = Path(CONFIG['base_dir'])  

    for i, sample_dir in enumerate(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        tissue_id = sample_dir.name
        experiment = METADATA[tissue_id]['condition']

        masks_dir = sample_dir / CONFIG['mask_dir']
        values_dir = sample_dir / CONFIG['values_dir']

        mask_files = [m_file for m_file in masks_dir.glob('*.*tif*')
            if m_file.name.lower().endswith(CONFIG['img_extensions'])]

        val_files = [v_file for v_file in values_dir.glob('*.*tif*')
            if v_file.name.lower().endswith(CONFIG['img_extensions'])]

        value_images = []
        for value_file in val_files:
            value_lazy = _lazy_load_image(value_file, res_level=0)
            val_basename = value_file.name.split('.')[0]
            val_name = val_basename.split('-')[-2]

            if val_name not in VAL_DICTS and value_lazy.ndim == 2:
                VAL_DICTS[val_name] = {
                    0: {
                        'slice_name': val_name
                    }
                }

            if "Collagen" in val_basename:
                dilations = CONFIG['col_dilations']
            else:
                dilations = CONFIG['int_dilations']

            value_images.append({
                'value_lazy': value_lazy,
                'val_basename': val_basename,
                'val_name': val_name,
                'dilations': dilations
            })

        for mask_file in tqdm.tqdm(mask_files[:], desc="Processing masks..."):
            mask_image = tifffile.imread(mask_file) > 0
            mask_basename = mask_file.name.split('.')[0]
            mask_name = mask_basename.split('-')[-2]

            if mask_name not in MASK_DICTS and mask_image.ndim == 2: 
                cluster_name, subgroup = _parse_mask_basename(mask_basename, CONFIG['subgroups'])
                MASK_DICTS[mask_name] = {
                    0 : {
                        'slice_name': cluster_name,
                        'unit': UNIT_DICT[cluster_name],
                        "group" : subgroup
                    }
                }

            for cached_value in value_images[:]:
                value_lazy = cached_value['value_lazy']
                val_basename = cached_value['val_basename']
                val_name = cached_value['val_name']
                dilations = cached_value['dilations']

                extractor.process_image(
                    mask_image=mask_image,
                    value_image=value_lazy,
                    mask_dictionary=MASK_DICTS[mask_name],
                    value_dictionary=VAL_DICTS[val_name],
                    dilations=dilations,
                    min_instance_size=CONFIG['min_instance_size'],
                    tissue_id=tissue_id,
                    mask_name=mask_name,
                    condition=experiment
                )

    df = extractor.return_dataframe()
    df.to_csv(r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/fib-output_long.csv", index=False)
    feature_matrix = df.pivot(
        index=['tissue_id', 'group', 'unit', 'condition', 'mask_name', 'centroid_y', 'centroid_x'], 
        columns='val_name', 
        values='value'
    ).reset_index()

    feature_matrix.to_csv(r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/fib-output.csv", index=False)


