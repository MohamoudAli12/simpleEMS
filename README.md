# simpleEMS
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL_v3_or_later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyPI Version](https://img.shields.io/pypi/v/project-name.svg)](https://pypi.org/project/project-name/)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)

simpleEMS is a python library built on top of [openEMS](https://openems.readthedocs.io/en/latest/) to make the design of antennas and other RF structures simpler.

## Motivation

This Python package provides an alternative to proprietary suites with expensive licenses or the restrictive "Student" versions such as CST, HFSS and Sonnet
which often impose stifling mesh cell limits and memory caps. By leveraging the openEMS FDTD engine, it offers a scalable framework for simulating, optimizing, and
visualizing complex RF designs. whether you are a professional engineer, a freelancer, or a dedicated hobbyist,
this library will enable you to design RF circuits easily.

## Key Features:

### **Antenna Design**
- Inset-Fed Patch Antenna Design
- Probe-Fed Patch Antenna Design

### **Simulation & Optimization**
- Parameter Sweep
- S11 Optimization

### **Network Parameters Calculation**
- S11 (Reflection Coefficient)
- VSWR (Voltage Standing Wave Ratio)
- Z11 (Input Impedance)
- Input Power

### **Visualization & Plotting**
- Geometry Structure Visualization
- S11 Plotting
- VSWR Plotting
- Smith Chart for S11
- Complex Impedance (Z11) Plotting
- 2D Radiation Pattern
- 2D Directivity Plot
- 3D Radiation Pattern, Gain & Power Plot

### **Field Dump**
- Field Dump for ParaView
  - E-Field (Time and Frequency Domain)
  - H-Field (Time and Frequency Domain)
  - Current Density (Time and Frequency Domain)

### **Export Options**
- Gerber Export
- STL Export
- Touchstone S1P Export

## Demo
[This example](./examples/simple_inset_fed_patch_antenna.py) creates an inset fed patch antenna and simulates the created model.

```{literalinclude} ../../examples/simple_inset_fed_patch_antenna.py
:language: python
```

Below is a visualisation of the model created by the above code

![patch_antenna_2_45GHz.png](./images/patch_antenna_2_45GHz.png)
The result of the simulation
![s11.png](./images/s11.png)

![smith_chart.png](./images/smith_chart.png)

![z11.png](./images/z11.png)

![vswr.png](./images/vswr.png)

![radiation_pattern.png](./images/radiation_polar.png)

![directivity_polar.png](./images/directivity_polar.png)

![3d_directivity.png](./images/3d_directivity.png)

![3d_gain.png](./images/3d_gain.png)

![3d_power.png](./images/3d_power_pattern.png)



## Installation

Refer to the installation instructions in the docs.

