| [**Main Page**](../README.md) | [**How to Use PyPRS**](How%20to%20Use%20PyPRS.md) | [**Output**](Output.md) | [**A Demo Application**](A%20Demo%20Application.md) |
# How to Use PyPRS
This section provides instructions for running PyPRS on a **single computer** and a **cluster**.

## 1 🖥️ Running PyPRS on a Single Computer <a name="h1"></a>
To run PyPRS on a single computer, users just need to execute the **`GUI.py`** file located in the `UserInterface` package in a Python environment. Once the file is executed, the **Graphical User Interface (GUI)** will launch. In the GUI, users can:
- **select a procedure**
- **configure input parameters**
- **upload required files**
- **run the procedure**

Below is a screenshot of the GUI:

  <img width="379.52" height="396.8" alt="image" src="https://github.com/user-attachments/assets/11314524-ebef-4662-b1dc-4b184b50c0db" />

### 1.1 Setting Up a Built-in Procedure
To use a built-in procedure (e.g., GSP, KT, PASS, or FBKT), follow these steps:

#### 1) Configuring Input Parameters
Each procedure requires specific input parameters.  For detailed explanations of the input parameters for each procedure, please go to <a name="IP" href="./Input Parameters.md">Input Parameters</a>.

#### 2) Uploading Required Files
In addition to configuring parameters, users must upload two files: one is the **alternatives infromation file** (a **`.txt`** file) and the other one is the **simulation function file** (a **`.py`** file). For detailed discussions about these two files, please go to <a name="UF" href="./Uploading Files.md">Uploading Files</a>.



### 1.2 Setting Up a Custom Procedure
To use a custom procedure, follow these steps:

#### 1) Configuring Input Parameters
Below is a screenshot of the GUI for the custom procedure:

  <img width="379.52" height="396.8" alt="image" src="https://github.com/user-attachments/assets/2a3ab001-4d6a-4873-bb6a-90c63e7696a4" />


When configuring a custom procedure, there are two default input parameters that need to be set:

- **`Number of Processors`**: Number of processors used to run the procedure.
- **`Repeat`**: Number of times the problem is repeatedly solved.

In addition to these two default input parameters, users can define other custom input parameters based on their procedure's requirements.

#### 2) Uploading Required Files

When using a custom procedure, users also need to upload the **alternatives infromation file** and the **simulation function file**. Besides these two files, users must upload a **procedure file** (a **`.py`** file). In this file, users must define a function named `custom_procedure`, which implements the computational steps for one run of the custom procedure.  For detailed discussions about the file, please go to <a href="./Procedure File.md" name="PF">Procedure File</a>.
## 2 🌐 Running PyPRS on a Cluster
Running PyPRS on a cluster follows a process similar to running it on a single computer, with additional steps to configure the cluster and ensure proper communication among the computers. Note that **deploying a Ray cluster is supported only on Linux**.
### 2.1 Setting Up the Ray Cluster
To establish a Ray cluster, follow these steps:

1. **Designate a Head Node**: Choose one computer as the head node to coordinate the cluster.
2. **Initialize the Head Node**: On the head node, run the following command in the terminal to start the Ray cluster:
```bush
ray start --head --port=6379
```
3. **Connect Worker Nodes**: On each of the other computers (worker nodes), run the following command to connect to the head node, replacing `HEAD_NODE_IP` with the Internet Protocol (IP) address of the head node:
```bush
ray start --address=HEAD_NODE_IP:6379
```
### 2.2 Running PyPRS on the Cluster
After setting up the Ray cluster, users can run PyPRS on the head node using the same process described in <a href="#h1">**Running PyPRS on a Single Computer**</a>. 
## 📖 Notes

PyPRS enables users to run procedures via scripts, alongside the GUI. This supports varied user needs and allows easy integration into automated workflows or complex computational pipelines. For more details, please go to <a href="Scripts for Invoking the Procedures in PyPRS.md">Scripts for Invoking the Procedures in PyPRS</a>.

<p align="right"><a href="./Output.md"> Proceed to Output</a></p>

