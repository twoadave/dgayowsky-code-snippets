'''D. J. Gayowsky 2024
Various cryptohack challenges.'''

################################################################
#IMPORTS

import base64
from Crypto.Util.number import *

################################################################

#General-Encoding-ASCII
def general_encoding_ascii():
    
    array = [99, 114, 121, 112, 116, 111, 123, 65, 83, 67, 73, 73, 95, 112, 114, 49, 110, 116, 52, 98, 108, 51, 125]
    flag = ''

    for i in range(len(array)):
        flag = flag + str(chr(array[i]))

    print(flag)
    return None
#general_encoding_ascii()

#General-Encoding-Hex
def general_encoding_hex():
    
    hexval = '63727970746f7b596f755f77696c6c5f62655f776f726b696e675f776974685f6865785f737472696e67735f615f6c6f747d'
    flag = bytes.fromhex(hexval)
    print(flag)

    return None
#general_encoding_hex()

#General-Encoding-Base64
def general_encoding_base64():

    hexstring = '72bca9b68fc16ac7beeb8f849dca1d8a783e8acf9679bf9269f7bf'
    bytestring = bytes.fromhex(hexstring)
    flag = base64.b64encode(bytestring)
    print(flag)

    return None
#general_encoding_base64()

#General-Encoding-Long
def general_encoding_long():
    flag = long_to_bytes(11515195063862318899931685488813747395775516287289682636499965282714637259206269)
    print(flag)
    return None
#general_encoding_long()