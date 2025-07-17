| [**Main Page**](../README.md) | [**How to Use PyPRS**](How%20to%20Use%20PyPRS.md) | [**Output**](Output.md) | [**A Demo Application**](A%20Demo%20Application.md) |
# The Throughput Maximization Problem

## 1 📝 Problem Description
The throughput maximization problem considers a flow line system with three stations, labeled as Stations 1, 2, and 3. There are infinitely many jobs waiting in front of Station 1, and each job moves sequentially through all three stations. The service times at Stations 1, 2, and 3 are independently drawn from exponential distributions with service rates $s_1$, $s_2$, and $s_3$ respectively. At Stations 2 and 3, there is finite buffer storage, denoted as $b_2$ and $b_3$ respectively. When the buffer at Station $i$, where $i = 2, 3$, is fully occupied, Station $i - 1$ is blocked and must hold the completed job until the job at Station $i$ is finished and released. The objective of this problem is to determine the optimal allocation of the service rate and the buffer to maximize the expected steady-state throughput of the flow line subject to constraints $s_1 + s_2 + s_3 = \mathcal{L}_1$,  $b_2 + b_3 = \mathcal{L}_2$, and $x=(s_1,s_2,s_3,b_2,b_3)\in Z$. Here, $\mathcal{L}_1$ and $\mathcal{L}_2$ represent problem-specific parameters that define the feasible solution set. The mathematical formulation of this problem is as follows:

<p align="center">$\max_{x} \mathbb{E}[f(x; \xi)]$</p>
<p align="center">$\text{s.t.} \quad s_1 + s_2 + s_3 = \mathcal{L}_1$</p>
<p align="center">$\quad\quad b_2 + b_3 = \mathcal{L}_2 $</p>
<p align="center">$\quad\quad\quad  x=(s_1,s_2,s_3,b_2,b_3)\in Z$,</p>

where $f(x; \xi)$ is the random throughput of the flow line. For every feasible solution, we obtain observations of $f(x; \xi)$ by running simulation experiments. For each simulation experiment, we warm up the system with 2,000 jobs. After 2,000 jobs are processed, we observe the throughput of the subsequent 50 jobs. In this demonstration, we let $\mathcal{L}_1=50$ and $\mathcal{L}_2=50$ resulting in 57,624 alternatives available in the problem. 



## 2 🔧 Using PyPRS to Solve the Problem

In this demo application, **all the built-in procedures and a custom procedure, namely the equal allocation procedure**, are applied to solve the throughput maximization problem.  The experiments are conducted on a workstation equiped with **two 24-core Intel(R) Xeon(R) Gold 6248R processors with 128 Gigabytes of memory, and Microsoft Windows 10 Pro operating system**. To use PyPRS, users need the **alternatives information file** and **simulation function file**. Download these files directly by clicking the following link:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Uploading%20Files.zip">Download Uploading Files</a> **换地方部署README后链接需更新**



*Note*: When the number of alternatives for a problem is very large, manually creating the **alternatives information file** can be time-consuming. In this problem, we use a <a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Generate%20Alternatives%20Information%20File.zip">Python function</a> to automatically generate the **alternatives information file**. Users can make slight modifications to this function to generate the file for other problems. **换地方部署README后链接需更新**






### 2.1 GSP
The screenshot below displays the PyPRS GUI during GSP execution:

**添加**


After using GSP to solve the problem, the results are summarized in the table below, which reports the Probability of Correct Selection (PCS), Wall-Clock Time, Total Simulation Time, Total Sample Size, Utilization, and the 95% confidence intervals for the five metrics. Note that when calculating the PCS, if the mean of the selected best alternative is within $\delta$ to that of the true best alternative, we count it as a correct selection.
| Metric                    | Value       | 95% Confidence Interval |
|---------------------------|-------------|-------------------------|
| PCS                       | [Value]  **修改**   | [Lower, Upper]   **修改**       |
| Wall-Clock Time (s)       | [Value]   **修改**  | [Lower, Upper] **修改**         |
| Total Simulation Time (s) | [Value]  **修改**   | [Lower, Upper]     **修改**     |
| Total Sample Size         | [Value]  **修改**   | [Lower, Upper]    **修改**      |
| Utilization (%)           | [Value]  **修改**   | [Lower, Upper]    **修改**      |

Users can download the summary and detailed results produced by PyPRS for GSP by clicking the link below:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Uploading%20Files.zip">Download Results for GSP</a> **修改**
### 2.2 The KT Procedure
The screenshot below displays the PyPRS GUI during the KT procedure execution:

**添加**


After using the KT procedure to solve this problem, the results are summarized in the table below, which reports the PCS, Wall-Clock Time, Total Simulation Time, Total Sample Size, Utilization, and the 95% confidence intervals for the five metrics. Note that when calculating the PCS, if the mean of the selected best alternative is within $\delta$ to that of the true best alternative, we count it as a correct selection.

| Metric                    | Value       | 95% Confidence Interval |
|---------------------------|-------------|-------------------------|
| PCS                       | [Value] **修改**    | [Lower, Upper]**修改**          |
| Wall-Clock Time (s)       | [Value] **修改**    | [Lower, Upper] **修改**         |
| Total Simulation Time (s) | [Value] **修改**    | [Lower, Upper] **修改**         |
| Total Sample Size         | [Value] **修改**    | [Lower, Upper] **修改**         |
| Utilization (%)           | [Value] **修改**    | [Lower, Upper] **修改**         |

Users can download the summary and detailed results produced by PyPRS for the KT procedure by clicking the link below:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Uploading%20Files.zip">Download Results for KT</a> **修改**
### 2.3 The PASS Procedure
The screenshot below displays the PyPRS GUI during the PASS procedure execution:

**添加**


After using the PASS to solve the problem, the results are summarized in the table below, which reports the average mean of the surviving alternatives, Wall-Clock Time, Total Simulation Time, Total Sample Size, Utilization, and the 95% confidence intervals for the five metrics. Note that for the PASS procedure, we report the average mean of the surviving alternatives instead of the PCS, as multiple alternatives may remain when the procedure terminates.

| Metric                              | Value       | 95% Confidence Interval |
|-------------------------------------|-------------|-------------------------|
| Average Mean of Surviving Alternatives | [Value] **修改** | [Lower, Upper]  **修改**        |
| Wall-Clock Time (s)                 | [Value] **修改**    | [Lower, Upper]  **修改**        |
| Total Simulation Time (s)           | [Value]**修改**     | [Lower, Upper]  **修改**        |
| Total Sample Size                   | [Value] **修改**    | [Lower, Upper]  **修改**        |
| Utilization (%)                     | [Value]**修改**     | [Lower, Upper]  **修改**        |

Users can download the summary and detailed results produced by PyPRS for the PASS procedure by clicking the link below:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Uploading%20Files.zip">Download Results for PASS</a> **修改**
### 2.4 The FBKT Procedure
The screenshot below displays the PyPRS GUI during the FBKT procedure execution:

**添加**


After using the FBKT procedure to solve this problem, the results are summarized in the table below, which reports the PCS, Wall-Clock Time, Total Simulation Time, Total Sample Size, Utilization, and the 95% confidence intervals for the five metrics. Note that when calculating the PCS, if the mean of the selected best alternative is within $\delta$ to that of the true best alternative, we count it as a correct selection, where $\delta$ is the same as that used in GSP and the KT procedure.

| Metric                    | Value       | 95% Confidence Interval |
|---------------------------|-------------|-------------------------|
| PCS                       | [Value]**修改**     | [Lower, Upper] **修改**         |
| Wall-Clock Time (s)       | [Value]**修改**     | [Lower, Upper] **修改**         |
| Total Simulation Time (s) | [Value]**修改**     | [Lower, Upper] **修改**         |
| Total Sample Size         | [Value] **修改**    | [Lower, Upper] **修改**         |
| Utilization (%)           | [Value] **修改**    | [Lower, Upper]  **修改**        |

Users can download the summary and detailed results produced by PyPRS for the PASS procedure by clicking the link below:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Uploading%20Files.zip">Download Results for FBKT</a> **修改**
### 2.5 The Custom Procedure

In this demo application, when implementing the custom procedure, we consider the equal allocation procedure. The procedure takes two additional input parameters, total sampling budget `N` and `Reference Seed`, in addition to `Repeat` and `Number of Processors`. The selection structure of the procedure is straightforward. At the start of the selection process, based on the total sampling budget set by the users, the procedure evenly allocates the simulation budget across all alternatives. Then, the alternatives are evenly distributed to all processors. Each processor generates simulation samples from the assigned alternatives, and the alternative with the highest sample mean is selected as the best. Users can directly download the procedure file for the equal allocation procedure by clicking the link below:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Equal%20Allocation.zip">Download Procedure File</a> **换地方部署README后链接需更新**


The screenshot below displays the PyPRS GUI during the equation allocation procedure execution:

**添加**



After using the equal allocation procedure to solve this problem, the results are summarized in the table below, which reports the PCS, Wall-Clock Time, Total Simulation Time, Total Sample Size, Utilization, and the 95% confidence intervals for the five metrics. Note that when calculating the PCS, if the mean of the selected best alternative is within $\delta$ to that of the true best alternative, we count it as a correct selection, where $\delta$ is the same as that used in GSP and the KT procedure.

| Metric                    | Value       | 95% Confidence Interval |
|---------------------------|-------------|-------------------------|
| PCS                       | [Value] **修改**    | [Lower, Upper]**修改**          |
| Wall-Clock Time (s)       | [Value]  **修改**   | [Lower, Upper]**修改**          |
| Total Simulation Time (s) | [Value] **修改**    | [Lower, Upper]**修改**          |
| Total Sample Size         | [Value] **修改**    | [Lower, Upper]  **修改**        |
| Utilization (%)           | [Value]**修改**     | [Lower, Upper]  **修改**        |

Users can download the summary results produced by PyPRS for the equal allocation procedure by clicking the link below:

<a href="https://raw.githubusercontent.com/biazhong/test/refs/heads/main/files/Uploading%20Files.zip">Download Results for the Equal Allocation Procedure</a> **修改**

