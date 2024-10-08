
import setuptools

#requirements = ["numpy", "scipy", "sklearn", "nibabel", "nilearn", "pandas", "statsmodels", "bctpy", "matplotlib", "h5py"]

reqs = ["bctpy==0.5.2", \
        "h5py==3.3.0", \
        "itkwidgets==0.32.1", \
        "matplotlib==3.4.3", \
        "nibabel==3.2.1", \
        "nilearn==0.8.1", \
        "nltools==0.4.5", \
        "numpy==1.21.3", \
        "pandas==1.3.2", \
        "panel==1.4.4", \
        "PyGSP @ git+https://github.com/epfl-lts2/pygsp@a38b41b84652d925deb6d94e9f42494cd90ddf68", \
        "pyvista==0.36.1", \
        "scikit-learn==0.24.2", \
        "scipy==1.7.1", \
        "seaborn==0.11.2", \
        "statsmodels==0.12.2"]

setuptools.setup(
    name="OCD_baseline",
    version="0.0.2",
    author="Sebastien Naze",
    author_email="sebastien.naze@gmail.com",
    description="OCD Analysis and Modeling",
    url="https://github.com/sebnaze/OCD-Analysis",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=reqs,
    include_package_data=True,
)
