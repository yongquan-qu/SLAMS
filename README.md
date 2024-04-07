# SLAMS: Score-based Latent Assimilation in Multimodal Setting

🎉🎉 __Accepted to CVPR2024 #EarthVision__ 🎉🎉

We recast data assimilation in multimodal setting using deep generative framework. In particular, we implement __latent score-based diffusion model__ where we project the heterogeneous states and observations into a __unified latent space__ where the forward and reverse conditional diffusion processes take place. Through varying ablation study, given coarse, noisy, and sparse conditioning inputs, we find our method to be robust and physically consistent.

📖: TBD


## Quickstart
1. Install dependencies using `pip` or `conda`
```
pip install -r requirements.txt
```

2. Run sample notebooks under `notebooks/` marked with `01_` prefix.
- `a`: Lorenz'63 system
- `b`: Kolmogorov fluid

## Full Experiments
In order to reproduce the results in the paper, we have to acquire the necessary data:
1. Process the in-situ data `python process_cpc.py`
2. Process the ex-situ data `python process_noaa.py`
3. Process the ERA5 data, follow quick instructions from https://leap-stc.github.io/ChaosBench/quickstart.html
4. Update `slams/config.py` field: `ERA_DATADIR = <YOUR_ERA5_DIR>` as instructed by the previous step 

4. All evaluations are summarized in a series of `notebooks/` marked with `02_` prefix.
    - `a`: Pixel-based data assimilation
    - `b`: Latent-based data assimilation NO observation (only background states)
    - `c`: Latent-based data assimilation with +1 observation (in-situ)
    - `d`: Latent-based data assimilation with +2 observation (in-situ + ex-situ)
    - `e`: Figures and tables generation


