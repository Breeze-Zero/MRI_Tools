## MRI_Tools

### Project Introduction

MRI_Tools is a basic toolkit for MRI data processing and reconstruction, which provides a series of MRI-related reconstruction applications, including parallel imaging, compressed sensing, coil sensitivity estimation, and so on.

### Main modules

1. **CS (Compressed Sensing)**: provides ` bart ` compression perception reconstruction method, and comes with a compression perception model based on deep learning ` CSANet `.
2. **CoilCombine**: It is used to compress multi-coil data and simulate single-coil data.
3. **Denoise**: Contains MRI data denoising methods based on `Noise2Noise` strategy, mainly applied to `M4Raw`dataset.
4. **PI (Parallel Imaging)**: Contains multiple parallel imaging methods and models such as `GRAPPA`, `MoDL`, `RAKI`, and `VarNet`.
5. **SME (Sensitivity Map Estimation)**: Provides `ESPIRIT` and Neural Network methods for coil sensitivity estimation
6. **SMS (Simultaneous Multi-Slice imaging)**: A range of tools for processing and reconstructing SMS data. Mainly, the `fastMRI `data are simulated into SMS data.
7. **mri_utils**: A suite of practical MRI tools including Pytorch/Numpy k-space data processing, conversion, simulation, and display.

### Installation

You need to install the Pytorch and Bart toolboxes yourself, and then call `pip install -r requirement.txt`

### Application Method

Each module has a specific Python file or Jupyter notebook, shows how to use the corresponding method. Users can refer to these documents to understand the details of each method using way, welcome friends have the idea to complement this part.

### **Dependency**

The toolkit uses many fastMRI, Direct, pygrappa and Bart's functions. When using Bart related functions need to import the Bart path.

### LICENSE

This project is distributed under the LICENSE provided in the License file.

### Contribution

The project currently only makes a preliminary sorting, welcome to contribute to the project. Play with the existing code, find bugs and improvements as you go, and then submit your Pull Request.

Thanks to fastMRI, Direct, pygrappa, and Bart for their great tools.
