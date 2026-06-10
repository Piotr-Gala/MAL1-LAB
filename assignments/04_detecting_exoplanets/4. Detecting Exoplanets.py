# %% [markdown]
# ## 4. Preprocessing and Performance: Detecting Exoplanets

# %% [markdown]
# **Objective**: Utilize data from the **K2 Kepler mission** (which concluded in 2018) to develop a machine learning model that assists in classifying celestial bodies and determining their exoplanet status. An exoplanet is defined as "A planet that orbits a star outside the solar system".
# 
# **Background**: The Kepler Mission was strategically devised to survey a segment of the Milky Way galaxy. Its primary goal was to identify Earth-sized or smaller planets situated in or near the habitable zone. This would further help in estimating the fraction of stars in our galaxy that might host such planets (_Nasa.gov, 2018_). The assignment itself is based almost completely on a previous student project from MAL 2022 submitted by Pavel Balan and Alex Vasilianov.
# 
# **Data Source**: The dataset is provided by the NASA Exoplanet Archive, NASA Exoplanet Science Institute [here](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi). There are several datasets available, all from differet periods, but we will use the __Cumulative KOI Data__ dataset.
# 
# The cumulative KOI (Kepler Objects of interest) table gathers information from the individual KOI activity tables that describe the current results of different searches of the Kepler light curves. The intent of the cumulative table is to provide the most accurate dispositions and stellar and planetary information for all KOIs in one place. All the information in this table has provenance in other KOI activity tables (_exoplanetarchive.ipac.caltech.edu, 2018_).
# 
# The data has been locally downloaded and saved as `exoplanet_dataset.csv`.
# 
# 
# #### Overall Instructions
# 1. Explore the dataset to understand the features and their distributions.
# 2. Preprocess the data, handling any missing values, outliers, or other anomalies.
# 3. Choose appropriate machine learning algorithms for the classification task.
# 4. Train and validate your model, ensuring to avoid overfitting.
# 5. Evaluate the model's performance using relevant metrics.
# 
# Below some guidelines are given but the assignment is relatively "free".
# 
# Best of luck with your analysis!
# 

# %% [markdown]
# ### 1. Explore

# %%
# Loading the data (change this if you want other var-names, etc.)
import pandas as pd

exoplanet_df = pd.read_csv('exoplanet_dataset.csv')

print(exoplanet_df.shape, "- 9564 rows with 49 features")

pd.set_option('display.max_columns', None)
exoplanet_df.head()

# %% [markdown]
# **COLUMN NAME | COLUMN DESCRIPTION [Data measurement unit type]**
# 
# COLUMN kepid:          KepID <br/>
# COLUMN kepoi_name:     KOI Name <br/>
# COLUMN kepler_name:    Kepler Name <br/>
# COLUMN koi_disposition: Exoplanet Archive Disposition <br/>
# COLUMN koi_pdisposition: Disposition Using Kepler Data <br/>
# COLUMN koi_score:      Disposition Score <br/>
# COLUMN koi_fpflag_nt:  Not Transit-Like False Positive Flag <br/>
# COLUMN koi_fpflag_ss:  Stellar Eclipse False Positive Flag <br/>
# COLUMN koi_fpflag_co:  Centroid Offset False Positive Flag <br/>
# COLUMN koi_fpflag_ec:  Ephemeris Match Indicates Contamination False Positive Flag <br/>
# COLUMN koi_period:     Orbital Period [days] <br/>
# COLUMN koi_period_err1: Orbital Period Upper Unc. [days] <br/>
# COLUMN koi_period_err2: Orbital Period Lower Unc. [days] <br/>
# COLUMN koi_time0bk:    Transit Epoch [BKJD] <br/>
# COLUMN koi_time0bk_err1: Transit Epoch Upper Unc. [BKJD] <br/>
# COLUMN koi_time0bk_err2: Transit Epoch Lower Unc. [BKJD] <br/>
# COLUMN koi_impact:     Impact Parameter <br/>
# COLUMN koi_impact_err1: Impact Parameter Upper Unc. <br/>
# COLUMN koi_impact_err2: Impact Parameter Lower Unc. <br/>
# COLUMN koi_duration:   Transit Duration [hrs] <br/>
# COLUMN koi_duration_err1: Transit Duration Upper Unc. [hrs] <br/>
# COLUMN koi_duration_err2: Transit Duration Lower Unc. [hrs] <br/>
# COLUMN koi_depth:      Transit Depth [ppm] <br/>
# COLUMN koi_depth_err1: Transit Depth Upper Unc. [ppm] <br/>
# COLUMN koi_depth_err2: Transit Depth Lower Unc. [ppm] <br/>
# COLUMN koi_prad:       Planetary Radius [Earth radii] <br/>
# COLUMN koi_prad_err1:  Planetary Radius Upper Unc. [Earth radii] <br/>
# COLUMN koi_prad_err2:  Planetary Radius Lower Unc. [Earth radii] <br/>
# COLUMN koi_teq:        Equilibrium Temperature [K] <br/>
# COLUMN koi_teq_err1:   Equilibrium Temperature Upper Unc. [K] <br/>
# COLUMN koi_teq_err2:   Equilibrium Temperature Lower Unc. [K] <br/>
# COLUMN koi_insol:      Insolation Flux [Earth flux] <br/>
# COLUMN koi_insol_err1: Insolation Flux Upper Unc. [Earth flux] <br/>
# COLUMN koi_insol_err2: Insolation Flux Lower Unc. [Earth flux] <br/>
# COLUMN koi_model_snr:  Transit Signal-to-Noise <br/>
# COLUMN koi_tce_plnt_num: TCE Planet Number <br/>
# COLUMN koi_tce_delivname: TCE Delivery <br/>
# COLUMN koi_steff:      Stellar Effective Temperature [K] <br/>
# COLUMN koi_steff_err1: Stellar Effective Temperature Upper Unc. [K] <br/>
# COLUMN koi_steff_err2: Stellar Effective Temperature Lower Unc. [K] <br/>
# COLUMN koi_slogg:      Stellar Surface Gravity [log10(cm/s^2)] <br/>
# COLUMN koi_slogg_err1: Stellar Surface Gravity Lower Unc. [log10(cm/s^2)] <br/>
# COLUMN koi_slogg_err2: Stellar Surface Gravity Lower Unc. [log10(cm/s^2)] <br/>
# COLUMN koi_srad:       Stellar Radius [Solar radii] <br/>
# COLUMN koi_srad_err1:  Stellar Radius Upper Unc. [Solar radii] <br/>
# COLUMN koi_srad_err2:  Stellar Radius Lower Unc. [Solar radii] <br/>
# COLUMN ra:             RA [decimal degrees] <br/>
# COLUMN dec:            Dec [decimal degrees] <br/>
# COLUMN koi_kepmag:     Kepler-band [mag] <br/>

# %%
# For an easier comprehension, we will rename the columns into their description.

exoplanet_df = exoplanet_df.rename(columns={'kepid':'KepID',
'kepoi_name':'KOIName',
'kepler_name':'KeplerName',
'koi_disposition':'ExoplanetArchiveDisposition',
'koi_pdisposition':'DispositionUsingKeplerData',
'koi_score':'DispositionScore',
'koi_fpflag_nt':'NotTransit-LikeFalsePositiveFlag',
'koi_fpflag_ss':'koi_fpflag_ss',
'koi_fpflag_co':'CentroidOffsetFalsePositiveFlag',
'koi_fpflag_ec':'EphemerisMatchIndicatesContaminationFalsePositiveFlag',
'koi_period':'OrbitalPeriod, days',
'koi_period_err1':'OrbitalPeriodUpperUnc, days',
'koi_period_err2':'OrbitalPeriodLowerUnc, days',
'koi_time0bk':'TransitEpoch, BKJD',
'koi_time0bk_err1':'TransitEpochUpperUnc, BKJD',
'koi_time0bk_err2':'TransitEpochLowerUnc, BKJD',
'koi_impact':'ImpactParamete',
'koi_impact_err1':'ImpactParameterUpperUnc',
'koi_impact_err2':'ImpactParameterLowerUnc',
'koi_duration':'TransitDuration, hrs',
'koi_duration_err1':'TransitDurationUpperUnc, hrs',
'koi_duration_err2':'TransitDurationLowerUnc, hrs',
'koi_depth':'TransitDepth, ppm',
'koi_insol':'InsolationFlux, Earthflux',
'koi_insol_err1':'InsolationFluxUpperUnc, Earthflux',
'koi_insol_err2':'InsolationFluxLowerUnc, Earthflux',
'koi_model_snr':'TransitSignal-to-Noise',
'koi_tce_plnt_num':'TCEPlanetNumber',
'koi_tce_delivname':'TCEDeliver',
'koi_steff':'StellarEffectiveTemperature, K',
'koi_steff_err1':'StellarEffectiveTemperatureUpperUnc, K',
'koi_steff_err2':'StellarEffectiveTemperatureLowerUnc, K',
'koi_depth_err1':'TransitDepthUpperUnc, ppm',
'koi_depth_err2':'TransitDepthLowerUnc, ppm',
'koi_prad':'PlanetaryRadius, Earthradii',
'koi_prad_err1':'PlanetaryRadiusUpperUnc, Earthradii',
'koi_prad_err2':'PlanetaryRadiusLowerUnc, Earthradii',
'koi_teq':'EquilibriumTemperature, K',
'koi_teq_err1':'EquilibriumTemperatureUpperUnc, K',
'koi_teq_err2':'EquilibriumTemperatureLowerUnc, K',
'koi_slogg':'StellarSurfaceGravity, log10(cm/s^2)',
'koi_slogg_err1':'StellarSurfaceGravityUpperUnc, log10(cm/s^2)',
'koi_slogg_err2':'StellarSurfaceGravityLowerUnc, log10(cm/s^2)',
'koi_srad':'StellarRadius, Solarradii',
'koi_srad_err1':'StellarRadiusUpperUnc, Solarradii',
'koi_srad_err2':'StellarRadiusLowerUnc, Solarradii',
'ra':'RA, decimaldegrees',
'dec':'Dec, decimaldegrees',
'koi_kepmag':'Kepler-band, mag'
})

# %% [markdown]
# **Updated data type value analysis**

# %%
exoplanet_df.info()

# %% [markdown]
# #### Description of some of the features
# 
# Given that some of the features might not be as straightforward going by their names, here are some descriptions for some of them, taken from the [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

# %% [markdown]
# `DispositionScore` - A value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition. The value is calculated from a Monte Carlo technique such that the score's value is equivalent to the fraction of iterations where the Robovetter yields a disposition of CANDIDATE.
# 
# `NotTransit-LikeFlag` A KOI whose light curve is not consistent with that of a transiting planet. This includes, but is not limited to, instrumental artifacts, non-eclipsing variable stars, and spurious (very low SNR) detections.	
# 
# `StellarEclipseFlag` A KOI that is observed to have a significant secondary event, transit shape, or out-of-eclipse variability, which indicates that the transit-like event is most likely caused by an eclipsing binary. However, self-luminous, hot Jupiters with a visible secondary eclipse will also have this flag set, but with a disposition of PC.
# 
# `Centroid Offset Flag` The source of the signal is from a nearby star, as inferred by measuring the centroid location of the image both in and out of transit, or by the strength of the transit signal in the target's outer (halo) pixels as compared to the transit signal from the pixels in the optimal (or core) aperture.
# 
# `EphemerisMatchIndicatesContaminationFlag` The KOI shares the same period and epoch as another object and is judged to be the result of flux contamination in the aperture or electronic crosstalk.
# 
# `Upper/LowerUnc` Uncertainties Columns(positive +)(negative -) aka - the error range for the columns. 
# 
# `TransitEpoch`	The time corresponding to the center of the first detected transit in Barycentric Julian Day (BJD).
# 
# `ImpactParameter`	The sky-projected distance between the center of the stellar disc and the center of the planet disc at conjunction, normalized by the stellar radius.
# 
# `TransitDepth` (parts per million)	The fraction of stellar flux lost at the minimum of the planetary transit. Transit depths are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris.
# 
# `InsolationFlux` [Earth flux]	Insolation flux is another way to give the equilibrium temperature. It depends on the stellar parameters (specifically the stellar radius and temperature), and on the semi-major axis of the planet. It's given in units relative to those measured for the Earth from the Sun.
# 
# `Equilibrium Temperature` (Kelvin)	Approximation for the temperature of the planet. The calculation of equilibrium temperature assumes a) thermodynamic equilibrium between the incident stellar flux and the radiated heat from the planet, b) a Bond albedo (the fraction of total power incident upon the planet scattered back into space) of 0.3, c) the planet and star are blackbodies, and d) the heat is evenly distributed between the day and night sides of the planet.
# 
# `RA` Right ascension (abbreviated RA; symbol α) is the angular distance of a particular point measured eastward along the celestial equator from the Sun at the March equinox to the (hour circle of the) point in question above the earth.
# 
# `Dec` declination (abbreviated dec; symbol δ) is one of the two angles that locate a point on the celestial sphere in the equatorial coordinate system, the other being hour angle. 

# %% [markdown]
# #### Target identification and modelling
# 
# Because this problem uses a supervised approach, we can determine the labels from the dataframe.
# We have the columns `DispositionUsingKeplerData` and `ExoplanetArchiveDisposition`. The first one holds values of either 'CANDIDATE' or 'FALSE POSITIVE'. The second one - 'CONFRIMED', 'CANDIDATE' or 'FALSE POSITIVE'. We will use `DispositionUsingKeplerData` as our only label, as we will be searching only for candidates. `ExoplanetArchiveDisposition` will still be used for data visualization however.

# %%
import seaborn as sns
sns.countplot(x = exoplanet_df['DispositionUsingKeplerData'])
print(exoplanet_df['DispositionUsingKeplerData'].value_counts())

# %%
sns.countplot(x= exoplanet_df['ExoplanetArchiveDisposition'])
print(exoplanet_df['ExoplanetArchiveDisposition'].value_counts())

# %% [markdown]
# #### Tasks

# %% [markdown]
# -  Explore missing values by finding the the precentage of missing values for each column - print as a dataframe
# -  Visualise the missing values of the columns with the 5 most percentage of missing value
# -  Identify potential outliers of the numeric features. This can be done in many ways but you should probably try to write some kind of script/loop that will iterate through the features and calculate the number of values outside [Q1 - 1.5IQR ; Q3 + 1.5IQR]. Don't remove them just yet but discuss what could be a good approach.
# -  Determine what to do with potential outlier: keep, replace or remove

# %% [markdown]
# ### 2. Feature Engineering
# 
# #### Tasks
# 
# -  Based only on the analysis above, you should be able to remove some columns (two of the columns will have 100% missing values). Remove these columns.
# -  Filter out (i.e. remove) any irrelevant columns (e.g. names, IDs, etc.) - there should be 4
# -  Remove rows with missing values, NaNs, nulls and/or infinite values - if you want, you may choose to impute instead
# -  If you you chose to remove or replace outliers do this now. If you chose to keep, move on
# -  Create a correlation matrix and discuss (use only numeric columns, perhaps make an extra dataset just with numeric values) - drop appropriate columns
# -  The values of `ExoplanetArchiveDisposition` and `ExoplanetArchiveDisposition` are categorical and if they are to act like labels, you should change them. Change them as follows: 'FALSE POSITIVE' values is assigned a numerical value of `0`, 'CANDIDATE' assigned `1`, and 'CONFIRMED' a `2`. Same assignment in both features. The easiest way is to create two new features, call them `KeplerDispositionStatus` and `ArchiveDispositionStatus`, and then drop the originalæ features. It will look something like this (the last couple of columns in the new dataframe):

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# #### Tasks continued
# -  Consider whether some features should be transformed (e.g. using log, square root etc.) and do this if found relevant
# -  Place `KeplerDispositionStatus` as the target and remove the following from the features: `DispositionScore`, `KeplerDispositionStatus`, `ArchiveDispositionStatus
# - Consider scaling your (numeric) data
# - You should now have two datasets, one with cleaned features and one with the target labels (1 for Candidate and 0 otherwise)

# %% [markdown]
# ### 3. Train, Test, Validation
# - Consider whether to use cross validation or not
# -  Consider which method to use to split the data and do the appropriate splits - if using CV still make a test set

# %% [markdown]
# ### 4. Models and Fine Tuning
# - Use classification algorithms to train 2 classification models:
#     1. Logistic Regression
#     2. Support Vector Machine
# - Fine tune the models either manually or using grid or random search

# %% [markdown]
# ### 5. Evaluate
# - Display the confusion matrix for both models
# - Evaluate the models using accuracy, precision, recall, and f1-score


