"""
The Config module (for specifying the fuzzing configurations)
"""
# Tool Description
__description__ = "Criticality-guided fuzzing method to generate multi-sensor dataset"

# Tool Version
__version__ = "1.0.0"

# Tool Name
__prog__ = "blindhunter"



__author__ = "alex"

ACTOR_POOL = ['vehicle', 'walker']

DEFAULT_ID = [{"ego": 0}, {"candidate": 1}]
