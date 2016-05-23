"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix 
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "/Users/admin/Downloads/map-2.osm.xml"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "St.": "Street",
            "Ave": "Avenue",
            "Rd.":"Road"
            }





def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)






def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def is_postal_code(elem):
    return (elem.attrib['k'] == "addr:postcode")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    postal_code = defaultdict(int)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                if is_postal_code(tag):
                    audit_postal_code(postal_code,tag.attrib['v'])
    osm_file.close()
    return [postal_code, street_types]

new_postal_code = 110001


def audit_postal_code(invalid_postal_codes,postal_code):
#checks if postal code have right length    
    if len(postal_code) != 6:
        invalid_postal_codes[postal_code] += 1

    

def update_postal_code(postal_code):
#checks if postal code have right length, if not replaces with 110001 default
    if len(postal_code) != 6:
    
        return new_postal_code


def update_name(name, mapping):
    name = name.strip()
    name = name.split(" ")
    
    better_last = mapping[name[len(name)-1]]
    name[len(name)-1] = better_last
    name = " ".join(name)
 
    return name

print audit(OSMFILE)