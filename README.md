# Monsoon Prediction Indian Subcontinent

## Forecasting trends In Indian Summer Monsoon Rainfall By Varying Architecture of Artificial Neural Networks(ANN)
### Reference

1. **Who**: Sahai, A.K., Soman, M.K. and Satyan, V., 2000. All India summer monsoon rainfall prediction using an artificial neural network. Climate dynamics, 16(4), pp.291-302.  

**What**: Prediction of Indian Monsoon Rainfall using the artificial neural network (ANN) technique with error- back-propagation algorithm to provide prediction (hindcast) of ISMR on monthly and seasonal time scales.  

**Important Papers They Have Referred**: 
    1. General circulation models (GCMs) have also been used to predict ISMR. A comprehensive study of the performance of thirty-three GCMs of the atmosphere by Sperber and Palmer (1996) has shown that the skill of the GCMs to simulate intraseasonal variations in ISMR is rather limited.
    2. A large number of parameters have been identified and they have tended to fall into four general categories (Krishna Kumar et al. 1995): (1) regional conditions, (2) El Nino- Southern Oscillation (ENSO) indicators, (3) cross-equatorial flow and (4) global/hemispheric conditions. Among these predictors, one or more are selected and linear regression, power regression or other statistical methods were developed for long-range forecasting of ISMR.
    3. The time-series analysis for prediction of ISMR, in which past values from ISMR seasonal mean time series are used to predict future values. Mooley and Parthasarathy (1984) analysed periodicity in 100 years of data and found two cycles (2.8 years and 14 years). Satyan (1988) has analysed 16 years of data using the phase space approach and found that a strange attractor of dimensionality around 5.1 exists and the system has 12 relevant degrees of freedom.
    4. Artificial neural networks (ANNs) have been extensively used for performing non-linear function approximation.  

**Motivation**: They have used ANNs with the presumption that ISMR is not only related to the previous seasonal mean but also to the previous monthly mean.

**Learning Algorithm**: 
    1. The network designed for the present study consists of four layers input (25 neurons which are previous five-year values from each time series of monthly mean ISMR values of June, July, August and September and the seasonal mean), output (1 neuron, the next year value from any one of the time series under consideration)and two hidden (2 and 4 neurons)
    2. Thus the total number of connections 69 (62 weights and 7 biases) is less than 85 (the number of training patterns) giving a reduced possibility of overtraining. Networks are trained separately for seasonal, June, July, August and September mean time series. Depending on the nature of the activation function two models were developed for each time series and 10 networks were trained, two for each time series
    3. 5 years and 5 features per year to help predict the rainfall.
    4. Specs: 25 input neurons, 2 hidden(2 and 4 neurons, respectively), and one output stating the predicted seasonal rainfall in mm.
    
**Evaluation**: Plots of predicted and observed.

2. **Who**: Singh, P. and Borah, B., 2013. Indian summer monsoon rainfall prediction using artificial neural network. Stochastic environmental research and risk assessment, 27(7), pp.1585-1599.

    **What**: Used feed-forward backpropagation neural network algorithm for ISMR forecasting.
    Based on this algorithm, they have proposed the five neural network architectures designated as BP1, BP2, . . .; BP5 using three layers of neurons (one input layer, one hidden layer and one output layer).

    **Motivation**: 1. Usually feedforward neural network (FFNN) and back-propagation neural network (BPNN) are used in ISMR forecasting.  
    2. Aksoy and Dahamsheh (2008)
    forecasted the precipitation for 1-month advance using the
    feed-forward back-propagation (FFBP), radial basis
    function (RBF) and generalizes regression (GR) neural
    networks. Later, all these three types of ANN models are
    compared with multiple linear regression (MLR), and FFBP is reported to be better than RBF and GR ANN including MLR. But in case of low precipitation region, RBF is better than FFBP.

    **Dataset**: 
    1. The time series data for 140 years (1871–2010) are divided into two parts as: (a) training set
    from the period 1871–1960, and (b) testing set from the
    period 1961–2010. Thus, there are (140*5 =) 700 entries
    of rainfall values in our model. As the previous year’s rainfall
    values are used for forecasting the next year, therefore predictions are available in the training set from the period
    1876–1960 and in the testing set from the period 1961–2010. 
    2. The arithmetic average of the rainfall values of the stations over the region, which also help in
    reducing the climatic noises or missing values present in data
    (especially daily data, which contain lots of noises and
    missing values). 

    **Model Used**:  

    1. Pearson correlation values between four months which reflect
    that rainfall are not pair-wise correlated. The correlation
    values for pair June–July (-0.0342) June–August (-0.0311), June–September (-0.0695), July–August (0.1005), July–September (0.2799) and August–September
    (0.2444) are very small, which also suggested that the
    relationships are not linear.
    2. ACFs for four months and seasonal time series.
    3. Normality of each time distribution using skewness and kurtosis test.
    4. An ANN based model is
    presented to predict ISMR of a given year using the observed
    time series data of the four months and seasonal. The model
    is developed based on supervised back-propagation neural
    network algorithm where the learning process aims to minimize the error rate between predicted output and the actual
    observation. 
    5. The performance of the model is assessed using various statistical parameters

    **Input/Output Specs**:
    1. Input consists of 5 nodes, time series pattern.

    **Evaluation**:  
    Parameters used for evaluation are Mean observed, Mean Predicted, SD observed, SD predicted, CC, RMSE, PP.

3. **Who**: Navone, H.D. and Ceccatto, H.A., 1994. Predicting Indian monsoon rainfall: a neural network approach. Climate Dynamics, 10(6-7), pp.305-312.

    **What**: The summer monsoon rainfall over India is predicted by using neural networks. These computational structures are used as a nonlinear method to correlate preseason predictors to rainfall data, and as an algorithm for reconstruction of the rainfall time-series intrinsic dynamics.

    1. They show that neural networks can be advantageously used in replacing standard linear statistical techniques. Instead of using computer-generated data, this will be exemplified on the problem of all-India SMR prediction because of its practical interest. 
    2. Secondly, and most important, by using neural networks they simply combine the deterministic and stochastic models of the SMR, producing a hybrid approach that performs almost 40% more accurately than the best standard methods using the same input data. A more complex model involving 16 predictors (Gowariker et al. 1989, 1991), have shown better skill than the methods referred to here. However, even with this much larger number of parameters, the performance of this model does not seem to reach the predictive power of the neural network. 
    
    **Predictors**:
    1.  Life cycle of the Southern Oscillation, and ii) seasonal transition of the midtropospheric circulation over India, show the most significant relationship with monsoon rainfall. 
    2.  The May surface resultant wind speed in a strategic area over the Indian
    Ocean is also relevant.
    
    **Neural Network**:1. Back Propagation Algorithm

    Test 1: Correlating the predictors )(1, X2 with the rainfall anomaly R by using neural networks. In this case a 2:2:1 (2 input units, 2 hidden units, and 1 output unit) net was used.  


4. **Who**: Guhathakurta, P., Rajeevan, M. and Thapliyal, V., 1999. Long range forecasting Indian summer monsoon rainfallby a hybrid principal component neural network model. Meteorology and atmospheric physics, 71(3-4), pp.255-266.
    
    **What**:
    1. An artificial neural network model by combining two different neural networks, one explaining assumed deterministic dynamics within the time series of Indian monsoon rainfall (Model I) and other using eight regional and global predictors (Model II)
    2. The model I has been developed by using the data of past 50 years (1901–50) and the data for recent period (1951–97) has been used for verification.
    3. The model II has been developed by using the 30 year (1958–87) data and the
    verification of this model has been carried out using the
    independent data of 10 year period (1988–97). In model II,
    instead of using eight parameters directly as inputs, we have
    carried out Principal Component Analysis (PCA) of the
    eight parameters with 30 years of data, 1958–87, and the
    first five principal components are included as input
    parameters.
    4. They have used the neural network technique to develop three different types of models for long range prediction of summer monsoon rainfall over India. In the first, we have used only time series data of monsoon rainfall as input to predict
    the monsoon rainfall for the future. This model (Model I) reconstructs the assumed deterministic dynamics of the monsoon rainfall data. 
    5. They have further used eight regional and global parameters as the predictors which are chosen by examining their physical linkage with the monsoon and their
    degree of relationship with the monsoon rainfall of India. Since some of these predictors are intercorrelated, we have carried out principal component analysis of these eight parameters and the first five principal components are taken as
    inputs to develop a principle component neural
    network model (Model II). 
    6. Finally these two models are combined by a two layer (without hidden layer) hybrid neural network model (Model III). The performance of the hybrid
    model (Model III) has been greatly improved over other two models.

    **Motivation**: Many papers proved that artificial neural networks perform the best on ISMR prediction.
    
    **Dataset**: 
    1. IITM dataset
    2.  Features for 8 feature neural network:
        *   Darwin Pressure Tendency (April–January), DPT 
        *   East Coast of India Minimum Temperature (March), ECT 
        *   Nino 3 Index Tendency (MAM-DJF), NI3l 
        *   NW Europe Mean Temperature (January), 
        *   NW India Pressure Anomaly (MAM), NWPA 
        *   NH Pressure Anomaly (Jan–April), NHPA 
        *   10 hPa Zonal Wind Anomaly at Balboa, 10HZA 
        *   Minimum Temperature of NW and Central India (May),MAYTM


5. **Who**: Venkatesan, C., Raskar, S.D., Tambe, S.S., Kulkarni, B.D. and Keshavamurty, R.N., 1997. Prediction of all India summer monsoon rainfall using error-back-propagation neural networks. Meteorology and Atmospheric Physics, 62(3-4), pp.225-240.
    **What**:Multilayered feedforward neural networks trained with the error-back-propagation (EBP) algorithm have been employed for predicting the seasonal monsoon rainfall over India. Three network models that use, respectively, 2, 3 and 10 input parameters which are known to significantly influence the Indian summer monsoon rainfall (ISMR) have been constructed and optimized.

    **Motivation**: Most systems encountered in the real-world are nonlinear and simple linear or linearized models cannot capture the essence of the underlying phenomena. Notwithstanding this, the usage of linear or linearized models continues for want of better methodologies to deal with the nonlinear systems. 
    
    **Dataset**: 
    1.  Based on their spatial domain, the predictors can be classified
    into four major groups as: (i) Indian region; (ii) Indian ocean region; (iii) E1 Nino; and (iv) global/hemispheric conditions.
    2.  All India (India taken as a single unit) summer
    monsoon (June-September) rainfall data for the
    period 1939-1994, prepared by area-weighting
    306 well-distributed (over the entire country)
    rain-guages has been taken from Parthasarathy
    et al. (1992, 1994). 
    3. Southern Oscillation Indicator: The seesaw oscillation in the sea level pressure between the Indian ocean and the east Pacific
    ocean over the near-equatorial latitudes is termed
    as southern oscillation.
    4. Surface Air Temperature: the mean surface temperatures at the west central
    India (WC1) stations during March, April and
    May have a high significant correlation (0.6) with
    the monsoon rainfall for the period 1951-1980.
    5. Bombay pressure Tendency: Bombay pressure, which is associated with the
    tropical circulation as well as the southern
    oscillation (Wright, 1975), is one of the leading
    predictors of the ISMR. Parthasarathy et al.
    (1991) showed that the seasonal mean sea level
    pressure tendency (MAM-DJF) for Bombay
    shows a significant correlation coefficient of
    -0.7 with the monsoon rainfall from the year
    1951 onwards.

 









