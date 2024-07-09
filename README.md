# Bromoform_Membrane

## Project Overview
This project offers sample codes for applying multiple analyses to trajectory data obtained from Molecular Dynamics (MD) simulations of bromoform molecules. Also included are the bromoform topology and parameter files in CHARMM format, available for download and use in MD simulation.


## Table of Contents
- [Installation](#installation)
- [Content](#content)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Citations](#citations)
- [License](#license)
- [Contact](#contact)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/CCCofficial/Bromoform_Membrane.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Bromoform_Membrane
   ```

## Content
1. Sample code:
   - `Bromoform0.2MPc_soapify_example.py`
     - Convert the xyz coordinate of bromoform molecules into SOAP vectors and apply dimensionality reduction and clustering analysis
   - `MSD_overlapping_window_Bromoform0.1MPcHundredFs.ipynb`
     - Apply overlapping time window method to calculate MSD and diffusion coefficient 
     
2. Topology and parameter files:
   - `bromoform_opls_charmm_gui_v02.rtf`
     - CHARMM-formatted bromoform topology file under the OPLS forcefield
   - `bromoform_charmm_gui_v02.prm`
     - CHARMM-formatted bromoform parameter file under the OPLS forcefield     

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests with your improvements.

## Acknowledgments
This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, Center for Cellular Construction.

Disclaimer: Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## Citations
- The bromform parameter file is generated based on the following paper:
  1. J. J. Karnes, I. Benjamin. "Deconstructing the Local Intermolecular Ordering and Dynamics of Liquid Chloroform and Bromoform." *Journal of Physical Chemistry B*, 2021, 125, 3629–3637. DOI: [10.1021/acs.jpcb.0c10407](https://doi.org/10.1021/acs.jpcb.0c10407).


- The SOAP analysis are carried out based on the following papers:
  1. A. P. Bartók, R. Kondor, G. Csányi. "On Representing Chemical Environments." *Physical Review B*, 2013, 87(18), 184115. DOI: [10.1103/PhysRevB.87.184115](https://doi.org/10.1103/PhysRevB.87.184115).

  2. A. Gardin, C. Perego, G. Doni, G. M. Pavan. "Classifying Soft Self-Assembled Materials via Unsupervised Machine Learning of Defects." *Communications Chemistry*, 2022, 5(1), 82. DOI: [https://doi.org/10.1038/s42004-022-00699-z](https://doi.org/10.1038/s42004-022-00699-z).


## License
This project is licensed under the MIT License.

## Contact
For any questions or issues, please open an issue in this repository or contact:
- Kevin Jose Cheng: kjcheng2@illinois.edu
- Jie Shi: shijie@ibm.com
- Taras Pogorelov: pogorelo@illinois.edu
- Sara Capponi: sara.capponi@ibm.com
