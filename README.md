
# Complex Neural Operator for Learning the Dynamics of Continuous Physical Systems

<p align="center">
<img src=".\fig\maincono (1).png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of CoNO.
</p>


## Codebase for Reproducibility

1. Install Python 3.8. For convenience, please go ahead and execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain experimental datasets from the following links (Download).


| Datasets       | Tasks                                    | Geometry        | Download Link                                                         |
| ------------- | --------------------------------------- | --------------- | ------------------------------------------------------------ |
| Elasticity-P  | Estimate material inner stress          | Point Cloud     | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Elasticity-G  | Estimate material inner stress          | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | Estimate material deformation over time | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Navier-Stokes | Predict future fluid velocity           | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| Darcy         | Estimate fluid pressure through medium  | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| AirFoil       | Estimate airﬂow velocity around airfoil | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | Estimate fluid velocity in a pipe       | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/.` You can reproduce the experiment results as the following examples:

```bash
bash scripts/elas_cono.sh # for Elasticity-P
bash scripts/elsa_interp_cono.sh # for Elasticity-G
bash scripts/plas_cono.sh # for Plasticity
bash scripts/ns_cono.sh # for Navier-Stokes
bash scripts/darcy_cono.sh # for Darcy
bash scripts/airfoil_cono.sh # for Airfoil
bash scripts/pipe_cono.sh # for Pipe
```
Note: You must change the argument `--data-path` in the above script files to your dataset path.

## Main Results

<p align="center">
<img src=".\fig\mainresults.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 2.</b> Main Results.
</p>

## Showcases

<p align="center">
<img src=".\fig\bigshowcase (1).png" height = "400" alt="" align=center />
<br><br>
<b>Figure 3.</b> Showcases.
</p>

## Citation

If you found the GitHub codebase useful, please cit following work:

@misc{tiwari2024conocomplexneuraloperator,
      title={CoNO: Complex Neural Operator for Continous Dynamical Physical Systems}, 
      author={Karn Tiwari and N M Anoop Krishnan and A P Prathosh},
      year={2024},
      eprint={2406.02597},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.02597}, 
}

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base or datasets on which we have built our code:

1) https://github.com/neuraloperator/neuraloperator

2) https://github.com/neuraloperator/Geo-FNO

3) https://github.com/tunakasif/torch-frft

5) https://github.com/soumickmj/pytorch-complex

6) https://github.com/thuml/Latent-Spectral-Models
