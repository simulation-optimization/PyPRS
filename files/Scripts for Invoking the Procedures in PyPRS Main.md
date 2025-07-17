| [**Main Page**](../README.md) | [**How to Use PyPRS**](How%20to%20Use%20PyPRS.md) | [**Output**](Output.md) | [**A Demo Application**](A%20Demo%20Application.md) |
#  Scripts for Invoking the Procedures in PyPRS
## 📋 Scripts for Invoking GSP
```python
from PyPRS.procedure import Procedure
	
gsp_procedure = Procedure("GSP")
gsp_procedure.set_CRNs() #Disable CRNs by commenting out this line

gsp_config = {
  "n0": <value>, # (int)
  "n1": <value>, # (int)
  "r_bar": <value>, # (int)
  "beta": <value>, # (int)
  "alpha1": <value>, # (float)
  "alpha2": <value>, # (float)
  "delta": <value>, # (float)
  "Number of Processors": <value>, # (int)
  "Repeat": <value>, # (int)
  "Reference Seed": <value>, # (list[int, int, int])
  "Alternatives Information File": "<path/to/file.txt>", # (str)
  "Simulation Function File": "<path/to/file.py>" # (str)
}	
gsp_result = gsp_procedure.run_procedure(gsp_config) 
# A list of dictionaries that record the final results.
```
## 📋 Scripts for Invoking the KT Procedure
```python
from PyPRS.procedure import Procedure
	
kt_procedure = Procedure("KT")
kt_procedure.set_CRNs() #Disable CRNs by commenting out this line

kt_config = {
    "alpha": <value>, # (float)
    "delta": <value>, # (float)
    "n0": <value>, # (int) 
    "g": <value>, # (int)
    "Number of Processors": <value>, # (int)
    "Repeat": <value>, # (int) 
    "Reference Seed": <value>, # (list[int, int, int]) 
    "Alternatives Information File": "<path/to/file.txt>", # (str) 
    "Simulation Function File": "<path/to/file.py>" # (str) 
}

kt_result = kt_procedure.run_procedure(kt_config) 
# A list of dictionaries that record the final results.
```
## 📋 Scripts for Invoking the PASS Procedure
```python
from PyPRS.procedure import Procedure
	
pass_procedure = Procedure("PASS")

pass_config = {
    "n0": <value>, # (int) 
    "Delta": <value>, # (int) 
    "Worker Elimination": <True\False>, # (bool)
    "c":  <value>, # (int)
    "Termination Sample Size":  <value>, # (int)
    "Number of Processors": <value>, # (int)
    "Repeat": <value>, # (int) 
    "Alternatives Information File": "<path/to/file.txt>", # (str)
    "Simulation Function File": "<path/to/file.py>" # (str)
}

pass_result = pass_procedure.run_procedure(pass_config)
# A list of dictionaries that record the final results.
```
## 📋 Scripts for Invoking the FBKT Procedure
```python
fbkt_procedure = Procedure("FBKT")
fbkt_procedure.set_CRNs() #Disable CRNs by commenting out this line

fbkt_config = {
    "N": <value>, # (int) 
    "n0": <value>, # (int) 
    "phi": <value>, # (int)
    "Number of Processors": <value>, # (int)
    "Repeat": <value>, # (int)
    "Reference Seed": <value>, # (list[int, int, int])
    "Alternatives Information File": "<path/to/file.txt>", # (str)
    "Simulation Function File": "<path/to/file.py>" # (str)
}

fbkt_result = fbkt_procedure.run_procedure(fbkt_config)
# A list of dictionaries that record the final results.
```
## 📋 Scripts for Invoking the Custom Procedure
```python
from PyPRS.procedure import Procedure
	
custom_procedure = Procedure("<path/to/file.py>") # (str) Path to the custom procedure file

custom_config = {
    # User-defined input parameters (optional, customizable)
    "param1": <value>,  # (type) Description of your custom parameter
    "param2": <value>,  # (type) Another user-defined parameter
    # ...

    # Required input parameters
    "Number of Processors": <value>,  # (int)
    "Repeat": <value>,  # (int)
    "Alternatives Information File": "<path/to/file.txt>",  # (str)
    "Simulation Function File": "<path/to/file.txt>"  # (str)
}

custom_result = custom_procedure.run_procedure(custom_config)
# A list of dictionaries that record the final results.
```

<a href="../README.md">Back to Main Page</a>


